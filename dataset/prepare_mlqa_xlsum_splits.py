"""
Prepare MLQA and XL-Sum splits for GRASP_LoRA.

- Downloads from Hugging Face (requires `datasets` and network).
- Creates train/val/micro_dev/test CSVs matching the split sizes in Table 1.
- Outputs to dataset/<dataset_name>/input/<prefix>.<split>.csv as expected by dataset_constants.py.
"""
import argparse
import csv
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from datasets import load_dataset

SPLIT_SIZES = {
    "mlqa": {
        "en": {"train": 3030, "val": 315, "micro_dev": 0, "test": 315},
        "ar": {"train": 50, "val": 50, "micro_dev": 16, "test": 100},
        "zh": {"train": 50, "val": 50, "micro_dev": 16, "test": 100},
    },
    "xlsum": {
        "en": {"train": 10000, "val": 1000, "micro_dev": 0, "test": 1000},
        "ar": {"train": 50, "val": 50, "micro_dev": 16, "test": 100},
        "zh": {"train": 50, "val": 50, "micro_dev": 16, "test": 100},
    },
}

MLQA_CONFIG = {"en": "en", "ar": "ar", "zh": "zh"}
XLSUM_CONFIG = {"en": "english", "ar": "arabic", "zh": "chinese_simplified"}


def ensure_dirs(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def sample_indices(n: int, sizes: Dict[str, int], seed: int) -> Dict[str, List[int]]:
    total_needed = sum(sizes.values())
    if n < total_needed:
        raise ValueError(f"Not enough examples: need {total_needed}, have {n}")
    rng = random.Random(seed)
    all_idx = list(range(n))
    rng.shuffle(all_idx)

    out = {}
    cursor = 0
    for split, count in sizes.items():
        out[split] = all_idx[cursor : cursor + count]
        cursor += count
    return out


def write_csv(rows: Iterable[Dict[str, str]], path: Path, fieldnames: List[str]):
    ensure_dirs(path)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def prepare_mlqa(lang: str, target_root: Path, seed: int):
    cfg = MLQA_CONFIG[lang]
    ds = load_dataset("mlqa", cfg)
    source = ds["train"] if "train" in ds else ds["validation"]
    sizes = SPLIT_SIZES["mlqa"][lang]

    idx_map = sample_indices(len(source), sizes, seed)

    def format_row(example):
        answers = example.get("answers", {})
        texts = answers.get("text", []) if isinstance(answers, dict) else []
        answers_text = "|".join(texts)
        return {
            "context": example["context"],
            "question": example["question"],
            "answers_text": answers_text,
        }

    for split, indices in idx_map.items():
        if sizes[split] == 0:
            continue
        split_rows = (format_row(source[i]) for i in indices)
        prefix = {"en": "en", "ar": "ar", "zh": "zh"}[lang]
        out_path = target_root / f"{prefix}.{split}.csv"
        write_csv(split_rows, out_path, fieldnames=["context", "question", "answers_text"])


def prepare_xlsum(lang: str, target_root: Path, seed: int):
    cfg = XLSUM_CONFIG[lang]
    ds = load_dataset("csebuetnlp/xlsum", cfg)
    source = ds["train"]
    sizes = SPLIT_SIZES["xlsum"][lang]

    idx_map = sample_indices(len(source), sizes, seed)

    def format_row(example):
        return {"article": example["text"], "summary": example["summary"]}

    for split, indices in idx_map.items():
        if sizes[split] == 0:
            continue
        split_rows = (format_row(source[i]) for i in indices)
        prefix = {"en": "en", "ar": "ar", "zh": "zh"}[lang]
        out_path = target_root / f"{prefix}.{split}.csv"
        write_csv(split_rows, out_path, fieldnames=["article", "summary"])


def main(output_dir: Optional[str] = None, seed: int = 42):
    base = Path(output_dir) if output_dir else Path(__file__).resolve().parent

    prepare_mlqa("en", base / "english_qa" / "input", seed)
    prepare_mlqa("ar", base / "arabic_qa" / "input", seed)
    prepare_mlqa("zh", base / "chinese_qa" / "input", seed)

    prepare_xlsum("en", base / "english_summary" / "input", seed)
    prepare_xlsum("ar", base / "arabic_summary" / "input", seed)
    prepare_xlsum("zh", base / "chinese_summary" / "input", seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=None, help="Optional base directory for outputs")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for deterministic sampling")
    args = parser.parse_args()
    main(args.output_dir, args.seed)
