#!/usr/bin/env bash
set -Eeuo pipefail
R=/root/autodl-tmp
exec 9>"$R/paired_full400_h1.lock"
flock -n 9 || exit 1
export HF_HOME=$R/hf_cache HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
for model in parent teacher31d; do
 for s in spatial object goal 10; do
  test ! -e "$R/outputs/full400_h1_${model}_${s}_seed1000" || { echo 'Refusing existing output'; exit 1; }
 done
done
for model in parent teacher31d; do
 path="$R/outputs/smolvla_libero_100k/checkpoints/100000/pretrained_model"
 extra=()
 if [[ "$model" == teacher31d ]]; then
  path="$R/models/lerobot_smolvla_libero_31d453f7"
  extra=('--rename_map={"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}')
 fi
 printf '%s full400_h1_start model=%s states0-9 seed1000 AMPfalse batch1 num_steps10\n' "$(date -Is)" "$model" >> "$R/PIPELINE_STATUS.txt"
 pids=()
 for s in spatial object goal 10; do
  /root/miniconda3/bin/python -m lerobot.scripts.lerobot_eval --policy.path="$path" --policy.device=cuda --policy.use_amp=false --policy.num_steps=10 --policy.n_action_steps=1 --env.type=libero --env.task="libero_$s" --env.init_state_offset=0 --eval.n_episodes=10 --eval.batch_size=1 --eval.use_async_envs=false --eval.recording=false "${extra[@]}" --output_dir="$R/outputs/full400_h1_${model}_${s}_seed1000" --job_name="full400_h1_${model}_${s}" --seed=1000 > "$R/logs/full400-h1-${model}-${s}.log" 2>&1 & pids+=("$!")
 done
 rc=0
 for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
 printf '%s full400_h1_exit model=%s rc=%s keep_server_on\n' "$(date -Is)" "$model" "$rc" >> "$R/PIPELINE_STATUS.txt"
 ((rc == 0)) || exit "$rc"
done
printf '%s paired_full400_h1_complete keep_server_on no_training\n' "$(date -Is)" >> "$R/PIPELINE_STATUS.txt"
