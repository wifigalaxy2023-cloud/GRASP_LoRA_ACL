import argparse
import sys

from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoTokenizer, LlamaForCausalLM

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "dataset"
if str(DATASET_PATH) not in sys.path:
    sys.path.append(str(DATASET_PATH))

from dataset_constants import DATASET_LISTS, DATASET_SPECS, LORA_PATH_DICT
from generation_utils import generate_and_save_summaries


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", choices=DATASET_LISTS, required=True)
    parser.add_argument("--lora_type", choices=LORA_PATH_DICT.keys(), required=True)
    parser.add_argument("--few_shot_num", type=int, default=50)

    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--start_id", type=int, default=0)
    parser.add_argument("--end_id", type=int, default=100)
    parser.add_argument("--output_root", type=str, default=str(REPO_ROOT / "merge/output/generations"))
    args = parser.parse_args()

    directory, prefix = DATASET_SPECS[args.dataset_name]
    lora_label = args.lora_type.replace("few", f"few{args.few_shot_num}")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    output_file_name = output_root / (
        f"{args.dataset_name}.{lora_label}.{args.start_id}-{args.end_id}.{args.split}.{prefix}.json"
    )

    base_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    peft_model_name = LORA_PATH_DICT[args.lora_type].format(few_shot_num=args.few_shot_num)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = LlamaForCausalLM.from_pretrained(base_model_name, device_map='auto', torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(base_model, peft_model_name)
    model.merge_and_unload()

    generate_and_save_summaries(
        tokenizer = tokenizer,
        model = model,
        dataset_name = args.dataset_name,
        split = args.split,
        output_file_name = str(output_file_name),
        start_id = args.start_id,
        end_id = args.end_id,
        device = torch.device("cuda"),
    )


if __name__ == "__main__":
    main()
