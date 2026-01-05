#! /bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH="$REPO_ROOT/configurations/src:$REPO_ROOT/dataset:${PYTHONPATH:-}"

few_shot_num=${FEW_SHOT_NUM:-0}
split=${SPLIT:-test}
start_id=${START_ID:-0}
end_id=${END_ID:-100}
output_root="${OUTPUT_ROOT:-$REPO_ROOT/merge/output/generations}"

run_generation () {
  local dataset_name="$1"
  local lora_type="$2"
  local target_root="$3"

  python "$REPO_ROOT/merge/generate_lora.py" \
    --dataset_name="$dataset_name" \
    --lora_type="$lora_type" \
    --few_shot_num="$few_shot_num" \
    --split="$split" \
    --start_id="$start_id" \
    --end_id="$end_id" \
    --output_root="$target_root"
}

run_generation "arabic_qa" "arabic_qa" "$output_root/arabic_qa"
run_generation "chinese_qa" "chinese_qa" "$output_root/chinese_qa"
