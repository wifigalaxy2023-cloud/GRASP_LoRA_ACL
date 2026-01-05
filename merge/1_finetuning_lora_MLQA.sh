#! /bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
PORT=${PORT:-29500}
BASE_MODEL=${BASE_MODEL:-"meta-llama/Meta-Llama-3-8B-Instruct"}
export PYTHONPATH="$REPO_ROOT/configurations/src:$REPO_ROOT/dataset:${PYTHONPATH:-}"

EPOCHS=${EPOCHS:-10}
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/merge/output}"

train_adapter () {
  local dataset_name=$1
  local out_dir="$OUTPUT_ROOT/${dataset_name}_adapter"
  mkdir -p "$out_dir"

  echo ">>> Training adapter (dataset=${dataset_name}, epochs=$EPOCHS, output=${out_dir})"
  torchrun \
    --nnodes=1 \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_port=$PORT \
    finetuning.py \
    --use_peft \
    --peft_method=lora \
    --enable_fsdp=True \
    --model_name="$BASE_MODEL" \
    --dataset=custom_whole_dataset \
    --dataset_name="$dataset_name" \
    --num_epochs=$EPOCHS \
    --batch_size_training=1 \
    --gradient_accumulation_steps=8 \
    --output_dir="$out_dir"
  echo ">>> Finished training adapter for ${dataset_name}."
}

train_adapter "english_qa"
train_adapter "chinese_qa"
train_adapter "arabic_qa"
