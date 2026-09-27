#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/root/autodl-tmp
DATA_ROOT="$ROOT/datasets/HuggingFaceVLA/libero"
BASE="$ROOT/outputs/smolvla_libero_100k/checkpoints/100000/pretrained_model"
OUT="$ROOT/outputs/smolvla_task0_finetune_3k"
SMOKE="$ROOT/outputs/smolvla_task0_finetune_smoke"
STATUS="$ROOT/PIPELINE_STATUS.txt"
LOG="$ROOT/logs/finetune-task0-3k.log"
DATA_REVISION=86958911c0f959db2bbbdb107eb3e17c5f9c798e
EPISODES='[0,18,22,33,58,85,88,105,107,114,121,125,129,157,167,170,190,207,211,231,233,235,236,247,249,257,264,267,295,301,307,309,315,323,343,346,362,367]'

export HF_HOME="$ROOT/hf_cache"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

run_train() {
  "$ROOT/venv-torch214/bin/python" -c 'import torch; torch.set_autocast_dtype("cuda", torch.bfloat16); from lerobot.scripts.lerobot_train import main; main()' "$@"
}

common=(
  --policy.path="$BASE"
  --policy.optimizer_lr=0.00001
  --policy.scheduler_warmup_steps=100
  --policy.scheduler_decay_steps=3000
  --policy.scheduler_decay_lr=0.000001
  --policy.use_amp=true
  --policy.push_to_hub=false
  --policy.device=cuda
  --dataset.repo_id=HuggingFaceVLA/libero
  --dataset.root="$DATA_ROOT"
  --dataset.revision="$DATA_REVISION"
  --dataset.episodes="$EPISODES"
  --dataset.return_uint8=true
  --batch_size=64
  --num_workers=16
  --persistent_workers=true
  --prefetch_factor=2
  --wandb.enable=false
  --env_eval_freq=0
  --log_freq=50
  --seed=1000
)

printf '%s finetune_task0_smoke_start episodes=38 lr=1e-5\n' "$(date -Iseconds)" >> "$STATUS"
run_train "${common[@]}" \
  --steps=2 \
  --save_checkpoint=true \
  --save_freq=2 \
  --output_dir="$SMOKE" \
  --job_name=smolvla_task0_finetune_smoke \
  > "$ROOT/logs/finetune-task0-smoke.log" 2>&1
printf '%s finetune_task0_smoke_complete\n' "$(date -Iseconds)" >> "$STATUS"

printf '%s finetune_task0_start episodes=38 steps=3000 save_freq=500 lr=1e-5 seed=1000\n' "$(date -Iseconds)" >> "$STATUS"
run_train "${common[@]}" \
  --steps=3000 \
  --save_checkpoint=true \
  --save_freq=500 \
  --output_dir="$OUT" \
  --job_name=smolvla_task0_finetune_3k \
  > "$LOG" 2>&1
printf '%s finetune_task0_complete steps=3000\n' "$(date -Iseconds)" >> "$STATUS"
