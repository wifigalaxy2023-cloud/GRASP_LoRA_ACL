#! /bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
PORT=${PORT:-29500}
export PYTHONPATH="$REPO_ROOT/configurations/src:$REPO_ROOT/dataset:${PYTHONPATH:-}"

allocator=${ALLOCATOR:-input_parameter_zero}          # input[grad]_parameter[module]_zero[init]
num_epochs=${EPOCHS:-10}
BATCH_SIZE_TRAINING=${BATCH_SIZE_TRAINING:-1}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}

DELETE_PERCENT_COMPAT="--delete_percent=0"
GRPO_P_MIN=${GRPO_P_MIN:-0.10}
GRPO_P_MAX=${GRPO_P_MAX:-0.80}
GRPO_P_INIT=${GRPO_P_INIT:-0.40}
GRPO_UPDATE_INTERVAL=${GRPO_UPDATE_INTERVAL:-10}  # K steps between policy updates
GRPO_GROUP_SIZE=${GRPO_GROUP_SIZE:-3}               # M candidates per round
GRPO_MICROVAL_SIZE=${GRPO_MICROVAL_SIZE:-16}        # shadow evaluation batch size
GRPO_POLICY_LR=${GRPO_POLICY_LR:-0.05}
GRPO_KL_COEFF=${GRPO_KL_COEFF:-0.05}
GRPO_ENTROPY_BONUS=${GRPO_ENTROPY_BONUS:-0.01}

GRPO_MAX_DELTA_P=${GRPO_MAX_DELTA_P:-0.10}          # cap adjustment per commit

GRPO_RANDOM_SEED=${GRPO_RANDOM_SEED:-42}

OUT_ROOT="$REPO_ROOT/merge/output/grpo"
RUN_TAG=${RUN_TAG:-}

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

GRPO_FLAGS=(
  --grpo_enable=true
  --grpo_p_min="$GRPO_P_MIN"
  --grpo_p_max="$GRPO_P_MAX"
  --grpo_p_init="$GRPO_P_INIT"
  --grpo_update_interval="$GRPO_UPDATE_INTERVAL"
  --grpo_group_size="$GRPO_GROUP_SIZE"
  --grpo_microval_size="$GRPO_MICROVAL_SIZE"
  --grpo_policy_lr="$GRPO_POLICY_LR"
  --grpo_kl_coeff="$GRPO_KL_COEFF"
  --grpo_entropy_bonus="$GRPO_ENTROPY_BONUS"
  --grpo_max_delta_p="$GRPO_MAX_DELTA_P"
  --grpo_seed="$GRPO_RANDOM_SEED"
)

run_job() {
  local dataset_name="$1"
  local merge_models="$2"

  RUN_NAME="${dataset_name}_grpo_p${GRPO_P_INIT}_k${GRPO_UPDATE_INTERVAL}g${GRPO_GROUP_SIZE}_pmin${GRPO_P_MIN}_pmax${GRPO_P_MAX}${RUN_TAG:+$RUN_TAG}"
  OUT_DIR="$OUT_ROOT/$RUN_NAME"
  LOG_DIR="$OUT_DIR/logs"
  LOG_PATH="$LOG_DIR/train.log"
  mkdir -p "$LOG_DIR"

  START_TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  RUN_TIMESTAMP=$(date -u +"%Y%m%dT%H%M%SZ")
  KL_TAG="kl${GRPO_KL_COEFF}"
  ENT_TAG="ent${GRPO_ENTROPY_BONUS}"
  RUN_SUFFIX="${RUN_TIMESTAMP}_${KL_TAG}_${ENT_TAG}"
  START_EPOCH=$(date +%s)
  TIMING_JSON="$OUT_ROOT/${RUN_NAME}_timing.json"
  TIMING_COPY="$OUT_ROOT/best_ratios/${RUN_NAME}_${RUN_SUFFIX}_timing.json"
  export RUN_NAME START_TIMESTAMP TIMING_JSON

  PYTHON_ENTRY="$SCRIPT_DIR/finetuning_merge_del.py"
  IFS=' ' read -r -a MERGE_MODELS_ARGS <<< "$merge_models"

  torchrun --nnodes=1 --nproc_per_node=1 --master_port=$PORT "$PYTHON_ENTRY" \
    --enable_fsdp \
    --batch_size_training="$BATCH_SIZE_TRAINING" \
    --gradient_accumulation_steps="$GRAD_ACCUM_STEPS" \
    --dataset=custom_whole_dataset \
    --dataset_name="$dataset_name" \
    --allocator="$allocator" \
    --num_epochs="$num_epochs" \
    "${MERGE_MODELS_ARGS[@]}" \
    $DELETE_PERCENT_COMPAT \
    --output_dir="$OUT_DIR" \
    "${GRPO_FLAGS[@]}" \
    |& tee "$LOG_PATH"

  TORCHRUN_STATUS=${PIPESTATUS[0]}
  if [[ $TORCHRUN_STATUS -ne 0 ]]; then
    echo "torchrun exited with status $TORCHRUN_STATUS" >&2
    exit $TORCHRUN_STATUS
  fi

  HISTORY_PATH="$OUT_DIR/grpo_history.json"
  BEST_ROOT="$OUT_ROOT/best_ratios"
  BEST_INSTANCE_DIR="$BEST_ROOT/${RUN_NAME}_${RUN_SUFFIX}"
  mkdir -p "$BEST_INSTANCE_DIR"
  BEST_JSON="$BEST_INSTANCE_DIR/${RUN_NAME}_best_ratio.json"
  HISTORY_COPY="$BEST_INSTANCE_DIR/${RUN_NAME}_grpo_history.json"
  LOG_COPY="$BEST_INSTANCE_DIR/${RUN_NAME}_training.log"

  if [[ -f "$HISTORY_PATH" ]]; then
    cp "$HISTORY_PATH" "$HISTORY_COPY"
  fi
  if [[ -f "$LOG_PATH" ]]; then
    cp "$LOG_PATH" "$LOG_COPY"
  fi

  export HISTORY_PATH BEST_JSON GRPO_P_INIT RUN_NAME

  python - <<'PY'
import json
import math
import os

history_path = os.environ.get("HISTORY_PATH", "")
best_json = os.environ.get("BEST_JSON", "")
init_ratio = float(os.environ.get("GRPO_P_INIT", "0.0"))
run_name = os.environ.get("RUN_NAME", "unknown_run")

best_ratio = init_ratio
best_reward = float("-inf")
best_loss = math.inf
history_records = []

if os.path.isfile(history_path):
    with open(history_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            history_records.append(record)
            val_reward = record.get("val_reward")
            val_loss = record.get("val_loss")
            if isinstance(val_reward, (int, float)):
                better = val_reward > best_reward or (math.isclose(val_reward, best_reward) and isinstance(val_loss, (int, float)) and val_loss < best_loss)
                if better:
                    best_reward = val_reward
                    best_loss = val_loss if isinstance(val_loss, (int, float)) else best_loss
                    best_ratio = record.get("p_current", best_ratio)
            trials = zip(record.get("p_trials", []), record.get("rewards", []), record.get("trial_losses", []))
            for ratio, reward, loss in trials:
                if isinstance(reward, (int, float)):
                    better = reward > best_reward or (math.isclose(reward, best_reward) and isinstance(loss, (int, float)) and loss < best_loss)
                    if better:
                        best_reward = reward
                        best_loss = loss if isinstance(loss, (int, float)) else best_loss
                        if isinstance(ratio, (int, float)):
                            best_ratio = ratio

result = {
    "run_name": run_name,
    "best_delete_ratio": float(best_ratio),
    "best_kept_fraction": float(1.0 - best_ratio),
    "best_reward": None if best_reward == float("-inf") else float(best_reward),
    "best_loss": None if math.isinf(best_loss) else float(best_loss),
    "initial_delete_ratio": float(init_ratio),
    "history_records": len(history_records),
}

with open(best_json, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)

print(f"Saved best ratio to {best_json}")
PY

  if [[ -d "$OUT_DIR" ]]; then
    rm -rf "$OUT_DIR"
  fi

  echo "Best delete ratio written to $BEST_JSON"
}

run_job "arabic_summary" "--english_summary --arabic_summary"

run_job "chinese_summary" "--english_summary --chinese_summary"
