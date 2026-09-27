#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/root/autodl-tmp
PYTHON=/root/miniconda3/bin/python
DATA_ROOT="$ROOT/datasets/local/smolvla_targeted_mix_v2_31d"
BASE="$ROOT/outputs/smolvla_libero_100k/checkpoints/100000/pretrained_model"
OUT="$ROOT/outputs/smolvla_targeted_finetune_v2_31d_2k"
SMOKE="$ROOT/outputs/smolvla_targeted_finetune_v2_31d_smoke"
STATUS="$ROOT/PIPELINE_STATUS.txt"
LOG="$ROOT/logs/targeted-finetune-v2-31d-2k.log"
exec 9>"$ROOT/targeted_finetune_v2.lock"
flock -n 9 || exit 1
trap 'printf "%s targeted_v2_failed line=%s\n" "$(date -Is)" "$LINENO" >> "$STATUS"' ERR
while kill -0 978563 2>/dev/null; do
  state=$(ps -o stat= -p 978563 || true)
  [[ "$state" == Z* ]] && break
  sleep 20
done
"$PYTHON" "$ROOT/validate_mix_v2.py"

export HF_HOME="$ROOT/hf_cache"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

run_train() {
  "$PYTHON" -c 'import torch; torch.set_autocast_dtype("cuda", torch.bfloat16); from lerobot.scripts.lerobot_train import main; main()' "$@"
}

common=(
  --policy.path="$BASE"
  --policy.optimizer_lr=0.000002
  --policy.scheduler_warmup_steps=50
  --policy.scheduler_decay_steps=2000
  --policy.scheduler_decay_lr=0.0000005
  --policy.use_amp=true
  --policy.push_to_hub=false
  --policy.device=cuda
  --dataset.repo_id=local/smolvla_targeted_mix_v2_31d
  --dataset.video_backend=pyav
  --dataset.root="$DATA_ROOT"
  --dataset.return_uint8=true
  --batch_size=64
  --num_workers=16
  --persistent_workers=true
  --prefetch_factor=2
  --wandb.enable=false
  --env_eval_freq=0
  --log_freq=50
  --seed=20260927
)

if [[ -e "$OUT" || -e "$SMOKE" ]]; then
  printf '%s targeted_finetune_refused_existing_output out=%s smoke=%s\n' "$(date -Is)" "$OUT" "$SMOKE" >> "$STATUS"
  exit 2
fi

printf '%s targeted_v2_smoke_start base=100k steps=2 lr=2e-6 dataset=targeted_mix_v2_31d\n' "$(date -Is)" >> "$STATUS"
run_train "${common[@]}" \
  --steps=2 \
  --save_checkpoint=true \
  --save_freq=2 \
  --output_dir="$SMOKE" \
  --job_name=smolvla_targeted_finetune_v2_31d_smoke \
  > "$ROOT/logs/targeted-finetune-v2-31d-smoke.log" 2>&1
printf '%s targeted_finetune_smoke_complete\n' "$(date -Is)" >> "$STATUS"

printf '%s targeted_finetune_start base=100k steps=2000 save_freq=250 lr=2e-6_to_5e-7 seed=20260927\n' "$(date -Is)" >> "$STATUS"
run_train "${common[@]}" \
  --steps=2000 \
  --save_checkpoint=true \
  --save_freq=250 \
  --output_dir="$OUT" \
  --job_name=smolvla_targeted_finetune_v2_31d_2k \
  > "$LOG" 2>&1
printf '%s targeted_finetune_complete steps=2000 out=%s\n' "$(date -Is)" "$OUT" >> "$STATUS"
