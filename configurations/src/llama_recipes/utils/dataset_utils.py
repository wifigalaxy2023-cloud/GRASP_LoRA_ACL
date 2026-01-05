import sys
from pathlib import Path

import datasets
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(REPO_ROOT / "dataset"))

from dataset_constants import DATASET_FILES, PROMPT_TEMPLATES  # noqa: E402


QA_DATASETS = {"english_qa", "arabic_qa", "chinese_aq"}


def tokenize_add_label(sample, tokenizer, prompt_template, truncate_prefix="\nSummary:"):
    prompt = tokenizer.encode(tokenizer.bos_token + prompt_template.format(article=sample["article"]), add_special_tokens=False)
    summary = tokenizer.encode(sample["summary"] + tokenizer.eos_token, add_special_tokens=False)

    if len(prompt) + len(summary) > 4096:
        prompt = prompt[: (4080 - len(summary))] + tokenizer.encode(truncate_prefix, add_special_tokens=False)

    sample = {
        "input_ids": prompt + summary,
        "attention_mask": [1] * (len(prompt) + len(summary)),
        "labels": [-100] * len(prompt) + summary,
    }
    return sample


def tokenize_qa(sample, tokenizer, prompt_template, truncate_prefix="\nAnswer:"):
    answer_text = sample.get("answers_text", "")
    answer = answer_text.split("|")[0] if answer_text else ""
    prompt = tokenizer.encode(
        tokenizer.bos_token + prompt_template.format(context=sample["context"], question=sample["question"]),
        add_special_tokens=False,
    )
    answer_ids = tokenizer.encode(answer + tokenizer.eos_token, add_special_tokens=False)

    if len(prompt) + len(answer_ids) > 4096:
        prompt = prompt[: (4080 - len(answer_ids))] + tokenizer.encode(truncate_prefix, add_special_tokens=False)

    sample = {
        "input_ids": prompt + answer_ids,
        "attention_mask": [1] * (len(prompt) + len(answer_ids)),
        "labels": [-100] * len(prompt) + answer_ids,
    }
    return sample


def _load_and_tokenize(dataset_name, tokenizer, split, prompt_templates=PROMPT_TEMPLATES):
    """Load a specific split and tokenize according to dataset type."""
    data_files = DATASET_FILES.get(dataset_name, {})
    if split not in data_files:
        raise KeyError(f"Split '{split}' not found for dataset '{dataset_name}'. Available: {list(data_files.keys())}")

    dataset = datasets.load_dataset("csv", data_files=data_files, split=split)
    if dataset_name in QA_DATASETS:
        dataset = dataset.map(
            tokenize_qa,
            fn_kwargs={
                "tokenizer": tokenizer,
                "prompt_template": prompt_templates[dataset_name],
            },
            remove_columns=list(dataset.features),
        )
    else:
        dataset = dataset.map(
            tokenize_add_label,
            fn_kwargs={
                "tokenizer": tokenizer,
                "prompt_template": prompt_templates[dataset_name],
            },
            remove_columns=list(dataset.features),
        )
    return dataset


def get_custom_whole_dataset(dataset_config, tokenizer, split: str):
    return _load_and_tokenize(dataset_config.dataset_name, tokenizer, split, PROMPT_TEMPLATES)


def get_custom_few_dataset(dataset_config, tokenizer, split: str):
    dataset = datasets.load_dataset("csv", data_files=DATASET_FILES[dataset_config.dataset_name], split=split)
    if dataset_config.example_num < len(dataset):
        dataset = dataset.train_test_split(test_size=dataset_config.example_num, seed=42)["test"]
    dataset = dataset.map(tokenize_add_label, fn_kwargs={
        "tokenizer": tokenizer,
        "prompt_template": PROMPT_TEMPLATES[dataset_config.dataset_name],
    }, remove_columns=list(dataset.features))
    return dataset


DATASET_PREPROC = {
    "custom_whole_dataset": get_custom_whole_dataset,
    "custom_few_dataset": get_custom_few_dataset,
}


def get_preprocessed_dataset(
    tokenizer, dataset_config, split: str = "train"
) -> torch.utils.data.Dataset:
    if not dataset_config.dataset in DATASET_PREPROC:
        raise NotImplementedError(f"{dataset_config.dataset} is not (yet) implemented")

    def get_split():
        return (
            dataset_config.train_split
            if split == "train"
            else dataset_config.test_split
        )

    return DATASET_PREPROC[dataset_config.dataset](
        dataset_config,
        tokenizer,
        get_split(),
    )


def get_tokenized_split(dataset_name: str, tokenizer, split: str):
    """Load and tokenize an arbitrary split (e.g., micro_dev) for a given dataset."""
    return _load_and_tokenize(dataset_name, tokenizer, split, PROMPT_TEMPLATES)
