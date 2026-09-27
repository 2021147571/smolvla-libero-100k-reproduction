#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/root/autodl-tmp
DATA_ROOT="$ROOT/datasets/HuggingFaceVLA/libero"
OUTPUT_ROOT="$ROOT/outputs"
LOG_ROOT="$ROOT/logs"
STATUS_FILE="$ROOT/PIPELINE_STATUS.txt"
DATA_REVISION=86958911c0f959db2bbbdb107eb3e17c5f9c798e

export HF_HOME="$ROOT/hf_cache"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false

# PyTorch's default CUDA autocast dtype is FP16, while SmolVLA's frozen VLM
# parameters are BF16. Force Accelerate to select native BF16 mixed precision;
# unlike FP16 this does not create a GradScaler for BF16 gradients.
run_train() {
  /root/autodl-tmp/venv-torch214/bin/python -c 'import torch; torch.set_autocast_dtype("cuda", torch.bfloat16); from lerobot.scripts.lerobot_train import main; main()' "$@"
}

status() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$STATUS_FILE"
}

status "pipeline_started"

while screen -ls 2>/dev/null | grep -q 'libero_download'; do
  status "waiting_for_libero_download"
  sleep 60
done

while screen -ls 2>/dev/null | grep -q 'vlm_download'; do
  status "waiting_for_vlm_download"
  sleep 60
done

if grep -qE 'Traceback|Error:' "$LOG_ROOT/libero-download.log"; then
  status "libero_download_failed"
  exit 20
fi

if grep -qE 'Traceback|Error:' "$LOG_ROOT/vlm-download.log"; then
  status "vlm_download_failed"
  exit 21
fi

test -f "$DATA_ROOT/meta/info.json"
status "downloads_verified dataset_size=$(du -sh "$DATA_ROOT" | awk '{print $1}')"

COMMON_ARGS=(
  --policy.type=smolvla
  --policy.load_vlm_weights=true
  --policy.freeze_vision_encoder=true
  --policy.train_expert_only=true
  --policy.train_state_proj=true
  --policy.use_amp=true
  --policy.push_to_hub=false
  --policy.device=cuda
  --dataset.repo_id=HuggingFaceVLA/libero
  --dataset.root="$DATA_ROOT"
  --dataset.revision="$DATA_REVISION"
  --dataset.return_uint8=true
  --num_workers=16
  --persistent_workers=true
  --prefetch_factor=2
  --wandb.enable=false
  --env_eval_freq=0
  --log_freq=100
  --seed=1000
)

SELECTED_BATCH=""
for batch in 64 32 16 8 4; do
  status "smoke_start batch_size=$batch"
  smoke_dir="$OUTPUT_ROOT/smoke_bf16_bs${batch}"
  if run_train \
      "${COMMON_ARGS[@]}" \
      --batch_size="$batch" \
      --steps=2 \
      --save_checkpoint=true \
      --save_freq=2 \
      --output_dir="$smoke_dir" \
      --job_name="smolvla_libero_smoke_bs${batch}" \
      > "$LOG_ROOT/smoke-bs${batch}.log" 2>&1; then
    SELECTED_BATCH="$batch"
    status "smoke_passed batch_size=$batch"
    break
  fi

  if grep -qiE 'out of memory|CUDA error|CUBLAS_STATUS_ALLOC_FAILED' "$LOG_ROOT/smoke-bs${batch}.log"; then
    status "smoke_oom batch_size=$batch"
  else
    status "smoke_failed_non_oom batch_size=$batch"
    exit 30
  fi
done

if [[ -z "$SELECTED_BATCH" ]]; then
  status "no_batch_size_fit"
  exit 31
fi

status "training_100k_start batch_size=$SELECTED_BATCH"

run_train \
  "${COMMON_ARGS[@]}" \
  --batch_size="$SELECTED_BATCH" \
  --steps=100000 \
  --save_checkpoint=true \
  --save_freq=10000 \
  --output_dir="$OUTPUT_ROOT/smolvla_libero_100k" \
  --job_name=smolvla_libero_100k \
  > "$LOG_ROOT/train-100k.log" 2>&1

status "training_100k_complete batch_size=$SELECTED_BATCH"
