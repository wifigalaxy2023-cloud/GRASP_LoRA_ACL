import os
import pickle
import sys

import torch
import torch.distributed._shard.checkpoint as dist_cp
from merge_models import LlamaForCausalLMLoRAMerge
from safetensors.torch import load_file
from transformers import LlamaConfig

sys.path += ["/workspace/dataset/"]

from dataset_constants import DATASET_SPECS, LORA_PATH_DICT


def get_lora_paths(kwargs: dict):
    lora_paths = []
    output_dir_lora_list = []
    lora_path_suffix = "/adapter_model.safetensors"

    lora_labels = [
        "xs_en",
        "xs_ja",
        "wl_en",
        "wl_ja",
        "bb_ja",
        "nj_ja",
        "mj_ja",
        "mm_en",
        "tl_en",
        "arabic_summary",
        "english_summary",
        "chinese_summary",
        "english_qa",
        "arabic_qa",
        "chinese_qa",
    ]
    for lora_label in lora_labels:
        value = kwargs.get(lora_label)
        if not value:
            continue

        adapter_root = (
            value
            if isinstance(value, str) and not isinstance(value, bool)
            else LORA_PATH_DICT[lora_label]
        )
        adapter_path = (
            adapter_root
            if adapter_root.endswith(lora_path_suffix)
            else adapter_root + lora_path_suffix
        )
        lora_paths.append(adapter_path)
        output_dir_lora_list.append(lora_label)

    ties_dir = kwargs.get("ties_dir")
    if ties_dir:
        adapter_path = os.path.join(ties_dir, "adapter_model.safetensors")
        if not os.path.isfile(adapter_path):
            raise FileNotFoundError(f"Expected adapter_model.safetensors in {ties_dir}")
        lora_paths.append(adapter_path)
        output_dir_lora_list.append(os.path.basename(os.path.normpath(ties_dir)))

    few_shot_nums = [5, 50, 100, 200]
    lora_prefix_labels = ["bb_ja", "nj_ja", "mj_ja", "mm_en", "tl_en", "arabic_summary", "english_summary"]
    for few_shot_num in few_shot_nums:
        for prefix in lora_prefix_labels:
            key = f"{prefix}_{few_shot_num}"
            if kwargs.get(key):
                lora_paths.append(LORA_PATH_DICT[f"{prefix}_few"].format(few_shot_num=few_shot_num) + lora_path_suffix)
                output_dir_lora_list.append(key)

    output_dir = kwargs.get("output_dir", "")
    if not output_dir:
        output_dir_dataset = ""
        if "dataset_name" in kwargs:
            output_dir_dataset = DATASET_SPECS.get(kwargs["dataset_name"], ("", ""))[0]

        output_dir_example_num = ""
        if "example_num" in kwargs:
            output_dir_example_num = f"ex{kwargs['example_num']}"

        output_dir_lora = "+".join(output_dir_lora_list)
        output_dir = f"/workspace/lora/{output_dir_dataset}/merge/{output_dir_lora}_{output_dir_example_num}/"

        if "allocator" in kwargs:
            output_dir += kwargs["allocator"]
        if "ipt_threshold" in kwargs:
            output_dir += f"/ipt_thresh{kwargs['ipt_threshold']}"
        if "delete_percent" in kwargs:
            output_dir += f"/del_percent{kwargs['delete_percent']}"

    return lora_paths, output_dir


def shape_model_init_weights(lora_paths, layer_num=32):
    state_dict_list = []
    for lora_path in lora_paths:
        state_dict_list.append(load_file(lora_path))

    q_lora_A_weights_list, q_lora_B_weights_list, v_lora_A_weights_list, v_lora_B_weights_list = [], [], [], []
    for layer_idx in range(layer_num):
        q_lora_A_weights = [state_dict[f"base_model.model.model.layers.{layer_idx}.self_attn.q_proj.lora_A.weight"] for state_dict in state_dict_list]
        q_lora_B_weights = [state_dict[f"base_model.model.model.layers.{layer_idx}.self_attn.q_proj.lora_B.weight"] for state_dict in state_dict_list]
        v_lora_A_weights = [state_dict[f"base_model.model.model.layers.{layer_idx}.self_attn.v_proj.lora_A.weight"] for state_dict in state_dict_list]
        v_lora_B_weights = [state_dict[f"base_model.model.model.layers.{layer_idx}.self_attn.v_proj.lora_B.weight"] for state_dict in state_dict_list]

        q_lora_A_weights_list.append(q_lora_A_weights)
        q_lora_B_weights_list.append(q_lora_B_weights)
        v_lora_A_weights_list.append(v_lora_A_weights)
        v_lora_B_weights_list.append(v_lora_B_weights)

    return q_lora_A_weights_list, q_lora_B_weights_list, v_lora_A_weights_list, v_lora_B_weights_list


def remove_fsdp_model(distcp_checkpoint_path):
    distcp_checkpoint_dir_files = os.listdir(distcp_checkpoint_path)
    removed_files = []
    for file in distcp_checkpoint_dir_files:
        if (".distcp" in file) or (".metadata" in file):
            file_path = os.path.join(distcp_checkpoint_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
            removed_files.append(file_path)

    return removed_files


def extract_lora_weights_list(model, lora_num):
    state_dict = model.state_dict()

    q_lora_A_weights_list, q_lora_B_weights_list, v_lora_A_weights_list, v_lora_B_weights_list = [], [], [], []
    for layer_idx in range(32):
        q_lora_A_weights = [state_dict[f"model.layers.{layer_idx}.self_attn.q_lora_A_weights.{lora_idx}"] for lora_idx in range(lora_num)]
        q_lora_B_weights = [state_dict[f"model.layers.{layer_idx}.self_attn.q_lora_B_weights.{lora_idx}"] for lora_idx in range(lora_num)]
        v_lora_A_weights = [state_dict[f"model.layers.{layer_idx}.self_attn.v_lora_A_weights.{lora_idx}"] for lora_idx in range(lora_num)]
        v_lora_B_weights = [state_dict[f"model.layers.{layer_idx}.self_attn.v_lora_B_weights.{lora_idx}"] for lora_idx in range(lora_num)]

        q_lora_A_weights_list.append(q_lora_A_weights)
        q_lora_B_weights_list.append(q_lora_B_weights)
        v_lora_A_weights_list.append(v_lora_A_weights)
        v_lora_B_weights_list.append(v_lora_B_weights)

    return q_lora_A_weights_list, q_lora_B_weights_list, v_lora_A_weights_list, v_lora_B_weights_list


def save_lora_weights_list(distcp_checkpoint_path, lora_paths):
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

    config = LlamaConfig.from_pretrained(model_name, torch_dtype=torch.float32)
    model = LlamaForCausalLMLoRAMerge.from_pretrained(model_name, device_map="cpu", torch_dtype=torch.float32, config=config)

    q_lora_A_weights_list, q_lora_B_weights_list, v_lora_A_weights_list, v_lora_B_weights_list = shape_model_init_weights(lora_paths)

    model.set_weights(
        q_lora_A_weights_list=q_lora_A_weights_list,
        q_lora_B_weights_list=q_lora_B_weights_list,
        v_lora_A_weights_list=v_lora_A_weights_list,
        v_lora_B_weights_list=v_lora_B_weights_list,
    )

    state_dict = {"model": model.state_dict()}
    dist_cp.load_state_dict(
        state_dict=state_dict,
        storage_reader=dist_cp.FileSystemReader(distcp_checkpoint_path),
        no_dist=True,
    )
    model.load_state_dict(state_dict["model"])

    q_lora_A_weights_list, q_lora_B_weights_list, v_lora_A_weights_list, v_lora_B_weights_list = extract_lora_weights_list(model, len(lora_paths))

    lora_weights_dict = {
        "q_lora_A_weights_list": q_lora_A_weights_list,
        "q_lora_B_weights_list": q_lora_B_weights_list,
        "v_lora_A_weights_list": v_lora_A_weights_list,
        "v_lora_B_weights_list": v_lora_B_weights_list,
    }
    with open(f"{distcp_checkpoint_path}/lora_weights_dict.pkl", "wb") as f:
        pickle.dump(lora_weights_dict, f)

    removed_files = remove_fsdp_model(distcp_checkpoint_path)
    print(f"Removed FSDP model: {removed_files}")

    return 0
