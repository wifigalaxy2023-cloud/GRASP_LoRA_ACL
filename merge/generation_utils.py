import json
import os
import sys
import warnings
from pathlib import Path

import datasets
import torch
from tqdm import tqdm
from transformers.utils import logging

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "dataset"
if str(DATASET_PATH) not in sys.path:
    sys.path.append(str(DATASET_PATH))

from dataset_constants import (DATASET_FILES, LORA_PATH_DICT, MAX_TOKENS_DICT,
                               PROMPT_TEMPLATES, SUMMARY_WORDS)

warnings.simplefilter(action='ignore', category=FutureWarning)
logging.set_verbosity_error()

def get_lora_name(args):
    lora_path = LORA_PATH_DICT[args.lora_type].format(few_shot_num=args.few_shot_num)
    return lora_path


def extract_output(output_str, en_summary_word="Summary:"):
    index_of_summary = output_str.rfind(en_summary_word)
    result = output_str[index_of_summary + len(en_summary_word):]
    result = result.replace("</s>", "").replace("\n", " ").strip()
    return result


def truncate_input_prompt(prompt, tokenizer, device, max_length=4000, en_summary_word="Summary:"):
    prompt = tokenizer(
        tokenizer.bos_token + tokenizer.decode(prompt["input_ids"][0][:max_length], skip_special_tokens=True) + "\n" + en_summary_word,
        add_special_tokens=False, return_tensors="pt"
    ).to(device)
    return prompt


def generate_and_save_summaries(model, tokenizer, dataset_name, split, output_file_name, start_id, end_id, device):
    dataset = datasets.load_dataset("csv", data_files=DATASET_FILES[dataset_name], split=split)
    dataset_length = len(dataset)
    if end_id > dataset_length:
        raise ValueError(f"end_id {end_id} exceeds dataset length {dataset_length} for split '{split}'.")

    selected_indices = list(range(start_id, end_id))
    dataset_subset = dataset.select(selected_indices)

    eos_token_id = tokenizer.eos_token_id
    pad_token_id = eos_token_id[0] if type(eos_token_id) == list else eos_token_id
    max_new_tokens = MAX_TOKENS_DICT[dataset_name]["max_new_tokens"]
    max_io_tokens = MAX_TOKENS_DICT[dataset_name]["max_io_tokens"]
    en_summary_word = SUMMARY_WORDS[dataset_name]

    qa_datasets = {"english_qa", "arabic_qa", "chinese_qa"}
    is_qa = dataset_name in qa_datasets
    output_key = "generated_answer" if is_qa else "generated_summary"

    output_records = []
    progress_desc = "Generating answers" if is_qa else "Generating summaries"
    for idx, sample in enumerate(tqdm(dataset_subset, desc=progress_desc)):
        prompt_template = PROMPT_TEMPLATES[dataset_name]
        if "article" in sample:
            prompt_str = tokenizer.bos_token + prompt_template.format(article=sample["article"])
        elif "context" in sample and "question" in sample:
            prompt_str = tokenizer.bos_token + prompt_template.format(
                context=sample["context"],
                question=sample["question"],
            )
        else:
            missing_keys = ", ".join(sample.keys())
            raise KeyError(f"Sample does not include required keys for prompt formatting. Available keys: {missing_keys}")
        prompt = tokenizer(prompt_str, add_special_tokens=False, return_tensors="pt").to(device)

        if len(prompt["input_ids"][0]) > max_io_tokens - max_new_tokens:
            prompt = truncate_input_prompt(
                prompt,
                tokenizer,
                device,
                max_length = max_io_tokens - max_new_tokens,
                en_summary_word = en_summary_word,
            )

        beam_outputs = model.generate(
            **prompt,
            max_new_tokens = max_new_tokens,
            num_beams = 1,
            use_cache = False,
            do_sample = False,
            pad_token_id = pad_token_id,
            eos_token_id = eos_token_id,
            top_p = 1.0,
            temperature = 1.0,
        )

        output_str = tokenizer.batch_decode(beam_outputs, skip_special_tokens=True)
        output_str = [extract_output(output, en_summary_word=en_summary_word) for output in output_str]
        generated_summary = output_str[0] if output_str else ""

        record = {key: sample[key] for key in sample}
        record[output_key] = generated_summary
        record["dataset_index"] = selected_indices[idx]
        output_records.append(record)

        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(output_file_name), exist_ok=True)
    with open(output_file_name, "w", encoding="utf-8") as f:
        json.dump(output_records, f, ensure_ascii=False, indent=2)
