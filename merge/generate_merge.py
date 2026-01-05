import argparse
import pickle
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, LlamaConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "dataset"
if str(DATASET_PATH) not in sys.path:
    sys.path.append(str(DATASET_PATH))

from dataset_constants import DATASET_LISTS, DATASET_SPECS
from generation_utils import generate_and_save_summaries
from merge_models import LlamaForCausalLMLoRAMerge


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", choices=DATASET_LISTS, required=True)
    parser.add_argument("--lora_dict_name", type=str, help="Directory name under <lora_root>/<dataset_dir>/merge/ that contains lora_weights_dict.pkl", required=False)

    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--start_id", type=int, default=0)
    parser.add_argument("--end_id", type=int, default=100)
    parser.add_argument("--output_root", type=str, default=str(REPO_ROOT / "merge/output/generations"))
    parser.add_argument("--lora_root", type=str, default="/workspace/lora")
    parser.add_argument("--lora_path", type=str, help="Direct path to directory containing lora_weights_dict.pkl", default="")
    args = parser.parse_args()

    directory, prefix = DATASET_SPECS[args.dataset_name]
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    config = LlamaConfig.from_pretrained(model_name, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLMLoRAMerge.from_pretrained(model_name, device_map="cpu", torch_dtype=torch.float16, config=config)

    if args.lora_path:
        lora_weights_dir = Path(args.lora_path)
    else:
        if not args.lora_dict_name:
            raise ValueError("Either --lora_path or --lora_dict_name must be provided.")
        lora_weights_dir = Path(args.lora_root) / directory / "merge" / args.lora_dict_name

    lora_label = args.lora_dict_name or lora_weights_dir.name
    output_file_path = output_root / (
        f"{args.dataset_name}.{lora_label}.{args.start_id}-{args.end_id}.{args.split}.{prefix}.json"
    )

    lora_weights_path = lora_weights_dir / "lora_weights_dict.pkl"
    if not lora_weights_path.is_file():
        raise FileNotFoundError(f"Missing LoRA weights at {lora_weights_path}")

    with open(lora_weights_path, "rb") as f:
        lora_dict = pickle.load(f)
    model.set_weights(
        q_lora_A_weights_list=lora_dict["q_lora_A_weights_list"],
        q_lora_B_weights_list=lora_dict["q_lora_B_weights_list"],
        v_lora_A_weights_list=lora_dict["v_lora_A_weights_list"],
        v_lora_B_weights_list=lora_dict["v_lora_B_weights_list"],
    )
    model.to(torch.device("cuda"))

    generate_and_save_summaries(
        tokenizer = tokenizer,
        model = model,
        dataset_name = args.dataset_name,
        split = args.split,
        output_file_name = str(output_file_path),
        start_id = args.start_id,
        end_id = args.end_id,
        device = torch.device("cuda"),
    )


if __name__ == "__main__":
    main()
