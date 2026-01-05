import os
import random
import sys

import fire
import numpy as np
import torch
import torch.optim as optim
from llama_recipes.configs import fsdp_config as FSDP_CONFIG
from llama_recipes.configs import train_config as TRAIN_CONFIG
from llama_recipes.policies import AnyPrecisionAdamW, apply_fsdp_checkpointing
from llama_recipes.utils import fsdp_auto_wrap_policy
from llama_recipes.utils.config_utils import (generate_dataset_config,
                                              get_dataloader_kwargs,
                                              update_config)
from llama_recipes.utils.dataset_utils import get_preprocessed_dataset
from llama_recipes.utils.train_utils import (clear_gpu_cache,
                                             freeze_transformer_layers,
                                             get_policies, print_model_size,
                                             setup, setup_environ_flags)
from llama_recipes.utils.train_utils_merge import print_and_save_log, train
from peft import prepare_model_for_kbit_training
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.optim.lr_scheduler import StepLR
from transformers import AutoTokenizer, LlamaConfig

sys.path += ["/workspace/merge/"]

from merge_models import LlamaDecoderLayerLoRAMerge, LlamaForCausalLMLoRAMerge
from merge_training_utils import (get_lora_paths, save_lora_weights_list,
                                  shape_model_init_weights)


def main(**kwargs):
    train_config, fsdp_config = TRAIN_CONFIG(), FSDP_CONFIG()
    update_config((train_config, fsdp_config), **kwargs)
    random.seed(train_config.seed)
    np.random.seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    torch.cuda.manual_seed_all(train_config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(train_config.seed)

    if train_config.enable_fsdp:
        setup()
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    if torch.distributed.is_initialized():
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)

    use_cache = False if train_config.enable_fsdp else None
    wandb_run = None


    torch_dtype = torch.float32
    llama_config = LlamaConfig.from_pretrained(train_config.model_name, torch_dtype=torch_dtype)
    llama_config.use_cache = use_cache

    model = LlamaForCausalLMLoRAMerge.from_pretrained(
        train_config.model_name,
        torch_dtype=torch_dtype,
        config=llama_config,
        load_in_8bit=True if train_config.quantization else None,
        device_map="auto" if train_config.quantization else None,
        attn_implementation="sdpa" if train_config.use_fast_kernels else None,
    )

    lora_paths, output_dir = get_lora_paths(kwargs)
    q_lora_A_weights_list, q_lora_B_weights_list, v_lora_A_weights_list, v_lora_B_weights_list = shape_model_init_weights(lora_paths)
    model.set_weights(
        q_lora_A_weights_list=q_lora_A_weights_list,
        q_lora_B_weights_list=q_lora_B_weights_list,
        v_lora_A_weights_list=v_lora_A_weights_list,
        v_lora_B_weights_list=v_lora_B_weights_list,
    )

    for param in model.parameters():
        param.requires_grad = False
    for layer_idx in range(32):
        for lora_idx in range(len(lora_paths)):
            model.model.layers[layer_idx].self_attn.q_lora_A_weights[lora_idx].requires_grad = True
            model.model.layers[layer_idx].self_attn.q_lora_B_weights[lora_idx].requires_grad = True
            model.model.layers[layer_idx].self_attn.v_lora_A_weights[lora_idx].requires_grad = True
            model.model.layers[layer_idx].self_attn.v_lora_B_weights[lora_idx].requires_grad = True

    kwargs["output_dir"] = output_dir
    setattr(train_config, "output_dir", output_dir)
    os.makedirs(output_dir, exist_ok=True)


    tokenizer = AutoTokenizer.from_pretrained(train_config.model_name if train_config.tokenizer_name is None else train_config.tokenizer_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        print("WARNING: Resizing the embedding matrix to match the tokenizer vocab size.")
        model.resize_token_embeddings(len(tokenizer))

    print_model_size(model, train_config, rank if train_config.enable_fsdp else 0)

    if train_config.quantization:
        model = prepare_model_for_kbit_training(model)

    if train_config.enable_fsdp and fsdp_config.pure_bf16:
        model.to(torch.bfloat16)


    hsdp_device_mesh = None
    if fsdp_config.hsdp and fsdp_config.sharding_strategy == ShardingStrategy.HYBRID_SHARD:
        hsdp_device_mesh = hsdp_device_mesh(replica_group_size=fsdp_config.replica_group_size, sharding_group_size=fsdp_config.sharding_group_size)
        print("HSDP device mesh is ready")

    if train_config.enable_fsdp:
        if not train_config.use_peft and train_config.freeze_layers:
            freeze_transformer_layers(model, train_config.num_freeze_layers)

        mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank)
        my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, LlamaDecoderLayerLoRAMerge)

        device_id = 0
        if torch.cuda.is_available():
            device_id = torch.cuda.current_device()

        model = FSDP(
            model,
            auto_wrap_policy= my_auto_wrapping_policy if train_config.use_peft else wrapping_policy,
            cpu_offload=CPUOffload(offload_params=True) if fsdp_config.fsdp_cpu_offload else None,
            mixed_precision=mixed_precision_policy if not fsdp_config.pure_bf16 else None,
            sharding_strategy=fsdp_config.sharding_strategy,
            device_mesh=hsdp_device_mesh,
            device_id=device_id,
            limit_all_gathers=True,
            sync_module_states=train_config.low_cpu_fsdp,
            param_init_fn=(lambda module: module.to_empty(device=torch.device("cuda"), recurse=False))
            if train_config.low_cpu_fsdp and rank != 0 else None,
            use_orig_params=True,
        )
        if fsdp_config.fsdp_activation_checkpointing:
            apply_fsdp_checkpointing(model)
    elif not train_config.quantization and not train_config.enable_fsdp:
        if torch.cuda.is_available():
            model.to("cuda")

    dataset_config = generate_dataset_config(train_config, kwargs)

    dataset_train = get_preprocessed_dataset(tokenizer, dataset_config, split="train")
    if not train_config.enable_fsdp or rank == 0:
        print(f"--> Training Set Length = {len(dataset_train)}")

    dataset_val = get_preprocessed_dataset(tokenizer, dataset_config, split="test")
    if not train_config.enable_fsdp or rank == 0:
        print(f"--> Validation Set Length = {len(dataset_val)}")

    train_dl_kwargs = get_dataloader_kwargs(train_config, dataset_train, tokenizer, "train")

    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
        **train_dl_kwargs,
    )

    eval_dataloader = None
    if train_config.run_validation:
        val_dl_kwargs = get_dataloader_kwargs(train_config, dataset_val, tokenizer, "val")

        eval_dataloader = torch.utils.data.DataLoader(
            dataset_val,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
            **val_dl_kwargs,
        )

    if fsdp_config.pure_bf16 and fsdp_config.optimizer == "anyprecision":
        optimizer = AnyPrecisionAdamW(
            model.parameters(),
            lr=train_config.lr,
            momentum_dtype=torch.bfloat16,
            variance_dtype=torch.bfloat16,
            use_kahan_summation=False,
            weight_decay=train_config.weight_decay,
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=train_config.lr,
            weight_decay=train_config.weight_decay,
        )
    scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)

    results = train(
        model,
        train_dataloader,
        eval_dataloader,
        tokenizer,
        optimizer,
        scheduler,
        train_config.gradient_accumulation_steps,
        train_config,
        fsdp_config if train_config.enable_fsdp else None,
        local_rank if train_config.enable_fsdp else None,
        rank if train_config.enable_fsdp else None,
        wandb_run,
    )
    if not train_config.enable_fsdp or rank==0:
        [print_and_save_log(f'Key: {k}, Value: {v}', train_config) for k, v in results.items()]

    if rank == 0:
        save_lora_weights_list(distcp_checkpoint_path=kwargs["output_dir"], lora_paths=lora_paths)

if __name__ == "__main__":
    fire.Fire(main)
