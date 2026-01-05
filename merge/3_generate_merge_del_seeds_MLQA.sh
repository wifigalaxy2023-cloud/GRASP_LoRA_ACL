#! /bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH="$REPO_ROOT/configurations/src:$REPO_ROOT/dataset:${PYTHONPATH:-}"

split=${SPLIT:-test}
start_id=${START_ID:-0}
end_id=${END_ID:-100}
output_root=${OUTPUT_ROOT:-$REPO_ROOT/merge/output/generations}
SEEDS=(${SEEDS:-42 1337 9001})
MERGE_ROOT=${MERGE_ROOT:-$REPO_ROOT/merge/output}

AR_DELETE_PERCENTS=(${AR_DELETE_PERCENTS:-})
CH_DELETE_PERCENTS=(${CH_DELETE_PERCENTS:-})

generate_with_merge () {
  local dataset_name="$1"
  local lora_path="$2"
  local target_root="$3"

  mkdir -p "$target_root"
  python "$SCRIPT_DIR/generate_merge.py" \
    --dataset_name="$dataset_name" \
    --split="$split" \
    --start_id="$start_id" \
    --end_id="$end_id" \
    --output_root="$target_root" \
    --lora_path="$lora_path"
}

for seed in "${SEEDS[@]}"; do
  for delete_percent in "${AR_DELETE_PERCENTS[@]}"; do
    generate_with_merge \
      "arabic_qa" \
      "$MERGE_ROOT/merge_eng_ar_qa_del/del${delete_percent}_seed${seed}" \
      "$output_root/arabic_qa_merge/del${delete_percent}_seed${seed}"
  done

  for delete_percent in "${CH_DELETE_PERCENTS[@]}"; do
    generate_with_merge \
      "chinese_qa" \
      "$MERGE_ROOT/merge_eng_ch_qa_del/del${delete_percent}_seed${seed}" \
      "$output_root/chinese_qa_merge/del${delete_percent}_seed${seed}"
  done
done
