#! /bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
PORT=${PORT:-29500}
export PYTHONPATH="$REPO_ROOT/configurations/src:$REPO_ROOT/dataset:${PYTHONPATH:-}"

ALLOCATOR=${ALLOCATOR:-input_parameter_zero}                  # input[grad]_parameter[module]_zero[init]
NUM_EPOCHS=${EPOCHS:-10}
SEEDS=(${SEEDS:-42 1337 9001})
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-8}
BATCH_SIZE_TRAINING=${BATCH_SIZE_TRAINING:-1}
RUN_NAME_SUFFIX=${RUN_NAME_SUFFIX:-""}
BASE_OUT_ROOT="${BASE_OUT_ROOT:-$REPO_ROOT/merge/output}"

AR_DELETE_PERCENTS=(${AR_DELETE_PERCENTS:-})
CH_DELETE_PERCENTS=(${CH_DELETE_PERCENTS:-})

START_TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
START_EPOCH=$(date +%s)
export RUN_NAME START_TIMESTAMP TIMING_JSON

cleanup() {
  set +e
  local exit_status=$1
  local end_epoch
  end_epoch=$(date +%s)
  local end_timestamp
  end_timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  local duration=$(( end_epoch - START_EPOCH ))
  mkdir -p "$(dirname "$TIMING_JSON")"
  END_TIMESTAMP="$end_timestamp" \
  DURATION_SECONDS="$duration" \
  EXIT_STATUS="$exit_status" \
  python - <<'PY'
import json
import os

timing_path = os.environ["TIMING_JSON"]
payload = {
    "run_name": os.environ.get("RUN_NAME", "unknown_run"),
    "start_time_utc": os.environ.get("START_TIMESTAMP"),
    "end_time_utc": os.environ.get("END_TIMESTAMP"),
    "duration_seconds": int(os.environ.get("DURATION_SECONDS", "0")),
    "exit_status": int(os.environ.get("EXIT_STATUS", "0")),
}
with open(timing_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
PY
  if [[ -f "$TIMING_JSON" ]]; then
    mkdir -p "$(dirname "$TIMING_COPY")"
    cp "$TIMING_JSON" "$TIMING_COPY"
  fi
  exit "$exit_status"
}
trap 'cleanup $?' EXIT

run_job() {
  local dataset_name="$1"
  local merge_models="$2"
  local out_subdir="$3"
  local delete_percent="$4"
  local seed="$5"

  local out_root="$BASE_OUT_ROOT/$out_subdir"
  local run_name="del${delete_percent}_seed${seed}${RUN_NAME_SUFFIX}"
  OUT_DIR="$out_root/$run_name"
  TIMING_JSON="$out_root/${run_name}_timing.json"
  TIMING_COPY="$out_root/best_ratios/${run_name}_timing.json"
  mkdir -p "$OUT_DIR"

  DELETE_PARAMETER="--delete_percent=${delete_percent}"
  OUTPUT_DIR_ARG="--output_dir=$OUT_DIR"
  PYTHON_ENTRY="$SCRIPT_DIR/finetuning_merge_del.py"
  IFS=' ' read -r -a MERGE_MODELS_ARGS <<< "$merge_models"

  torchrun --nnodes=1 --nproc_per_node=1 --master_port=$PORT "$PYTHON_ENTRY" \
    --enable_fsdp \
    --batch_size_training="$BATCH_SIZE_TRAINING" \
    --gradient_accumulation_steps="$GRAD_ACCUM_STEPS" \
    --dataset=custom_whole_dataset \
    --dataset_name="$dataset_name" \
    --allocator="$ALLOCATOR" \
    --num_epochs="$NUM_EPOCHS" \
    "${MERGE_MODELS_ARGS[@]}" \
    "$DELETE_PARAMETER" \
    "$OUTPUT_DIR_ARG" \
    --seed="$seed"
}

for seed in "${SEEDS[@]}"; do
  for delete_percent in "${AR_DELETE_PERCENTS[@]}"; do
    run_job "arabic_summary" "--english_summary --arabic_summary" "merge_eng_ar_summary_del" "$delete_percent" "$seed"
  done
done

for seed in "${SEEDS[@]}"; do
  for delete_percent in "${CH_DELETE_PERCENTS[@]}"; do
    run_job "chinese_summary" "--english_summary --chinese_summary" "merge_eng_ch_summary_del" "$delete_percent" "$seed"
  done
done
