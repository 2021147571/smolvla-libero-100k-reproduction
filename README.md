# SmolVLA LIBERO 100k Reproduction

A personal archive of reproduction experiments. Model weights: [laroi0124/SmolVLA_100k_test](https://huggingface.co/laroi0124/SmolVLA_100k_test).

## Model Origins

| Model | How it was trained | Status |
|---|---|---|
| **100k parent model** | Initialized the VLM from `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`, not from a LIBERO teacher checkpoint. Froze the VLM and trained the action expert and state projection on `HuggingFaceVLA/libero` for 100,000 steps, with batch size 64 and seed 1000. | Main preserved and published model |
| task0 child model | Started from the 100k parent and used 38 successful demonstrations of the target task. Trained for another 3,000 steps, batch size 64, learning rate 1e-5 → 1e-6, saving every 500 steps. | Early success-demo-only experiment; not presented as a final improved model |
| Targeted fine-tuning v1 | Early teacher-guided targeted fine-tuning and child-checkpoint selection. | Abandoned; original files remain archived and should not be resumed |
| Targeted fine-tuning v2 (31d teacher) | Started from the parent, sampling teacher correction trajectories, target-task successful demonstrations, and replay across all short tasks at a 25%/25%/50% ratio. Trained for 2,000 steps, batch size 64, learning rate 2e-6 → 5e-7, saving every 250 steps for eight child checkpoints. | Failed experiment: new dataset statistics replaced the parent's normalization statistics. Not evidence of a successful improvement |
| v3 fixed-stats smoke test | Preserved the parent's normalization statistics and ran a two-step smoke test to validate the training path. | Debugging only; not a completed fine-tuning run |
| Official teacher `lerobot/smolvla_libero@31d453f7` | Downloaded public checkpoint whose saved training configuration specifies 25k steps; not trained in this project. | Comparison baseline; not confirmed to be the exact checkpoint behind the paper's results |
| 2.2B teacher candidate | Downloaded the 100000 checkpoint from `HuggingFaceVLA/smolvla_libero_ckpts` and performed compatibility checks. | Paused by the user; not presented as a fully evaluated or validated teacher |

## Parent Model Configuration

- Dataset revision: `86958911c0f959db2bbbdb107eb3e17c5f9c798e`.
- AdamW, initial learning rate `1e-4`, 1,000 warmup steps, scheduler decay setting of 30,000 steps, minimum learning rate `2.5e-6`; total training length remains 100,000 steps.
- Two image inputs, 8-dimensional state, 7-dimensional action; action chunk length 50 and 10 flow-matching integration steps.
- **Executing 1, 10, or 50 actions before observing again is an evaluation setting, not three separately trained parent models.** The original exported configuration defaults to 50 action steps.

## Completed Evaluations

| Protocol | Spatial | Object | Goal | Long | Total |
|---|---:|---:|---:|---:|---:|
| Parent, observe again after every 1 action | 87/100 | 83/100 | 82/100 | 53/100 | 305/400; short tasks 252/300 |
| Teacher 31d, observe again after every 1 action | 78/100 | 74/100 | 76/100 | 57/100 | 285/400; short tasks 228/300 |
| Parent, observe again after every 10 actions | 84/100 | 95/100 | 89/100 | Not evaluated | 268/300 |
| Teacher 31d, observe again after every 10 actions | 75/100 | 88/100 | 87/100 | Not evaluated | 250/300 |
| Parent, observe again after every 50 actions (early evaluation) | 69/100 | 75/100 | 71/100 | Not evaluated | 215/300 |

These are observed results in this project's environment, not official paper results. The evaluations in this table use 10 initial states per task (0–9), seed 1000, batch size 1, AMP disabled, and 10 flow integration steps. Compare the parent and teacher only at the same action execution interval. No claim of 300/300 success is made.

## Files and Limitations

- `results/`: evaluation outputs and summaries.
- `scripts/`: archived experiment scripts containing machine-specific paths that must be adjusted before use; not a one-click setup package. Abandoned fine-tuning scripts are retained only as historical records.
- `docs/`: source audits and experiment notes.
- Passwords, SSH keys, connection credentials, raw sensitive logs, dataset copies, and third-party teacher weights are excluded.
- The evaluation environment is not guaranteed to match the paper exactly. Simulator versions, randomness, and previously inspected test states may affect interpretation. Existing results do not establish that training and test states are disjoint.
- Abandoned child-model weights remain in the original experiment storage and are not uploaded to this code repository as recommended models.

Upstream projects: [LeRobot](https://github.com/huggingface/lerobot), [SmolVLA](https://arxiv.org/abs/2506.01844), and [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO). This is a personal reproduction archive, not an official release.
