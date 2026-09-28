# AnyCamVLA + frozen SmolVLA: small Medium pilot

No policy training. Official LVSM reconstructs the original external camera view
from the perturbed external view and unchanged wrist view. Only the external
policy image is replaced; the original 100k parent weights are unchanged.

## Results (2026-09-29)

Each row is **one task**, not a complete suite. Five initial states per task.

| Task (suite-local ID 0) | Original | Medium | Medium + LVSM |
|---|---:|---:|---:|
| Spatial: black bowl between plate and ramekin onto plate | 4/5 | 4/5 | 4/5 |
| Object: alphabet soup into basket | 4/5 | 1/5 | 5/5 |
| Goal: open middle drawer | 5/5 | 5/5 | 5/5 |
| Total | **13/15** | **10/15** | **14/15** |

Compared with Medium alone: 5 paired gains, 1 loss, net +4. LVSM run took
501.4 seconds on RTX4090. This is an integration pilot, **not Short-300**, not
a paper reproduction or evidence of general superiority over native views.
These states have already been inspected; they are not a fresh held-out test.
Per-episode evidence is in `results/`; no seeds or tasks were changed between groups.

## Fixed protocol

- Spatial0/Object0/Goal0; states30–34; seed1000+state.
- Batch1, action execution horizon1, flow integration10, policy AMPfalse.
- Official `combined_medium`: radius+0.1m, azimuth+30deg, elevation+5deg;
  camera looks at its original ray/workspace-plane intersection. Reference
  heights (official robot_base_z): Spatial0.912m, Object0m, Goal0.912m.
- Native360px images are vertically flipped for LVSM, whose preprocessing
  rescales images/intrinsics to256px. Output is flipped back and bilinearly
  resized to360px. Native policy preprocessing then runs unchanged.
- LVSM uses BF16. Its RNG use is isolated from policy sampling. Ground-truth
  restored-view images are saved for diagnostics only, never fed to the policy.
- Parent SHA256: `e72a92a681a70db19bf2fb685574156641d0deb33b0d8b173c20d32fdb85b11c`.

## Sources and dependencies

- [AnyCamVLA](https://github.com/heo0224/AnyCamVLA), commit
  `93836230197c563ce958dc25ec67eda6c176a444`.
- [Official LVSM weights](https://huggingface.co/heo0224/AnyCamVLA-LVSM):
  `ckpt_0000000000020000.pt`, SHA256
  `ad17d721477fb1fd845e2bae0659072ff31f7470aeb24c6deec91866e5658640`.
- Existing working LeRobot/LIBERO environment: Python3.12, Torch2.8.0+cu128,
  LeRobot0.6.1, MuJoCo3.8.1, robosuite1.4. Additional environment used
  xformers0.0.32.post2, lpips0.1.4, easydict, omegaconf, jaxtyping.
- Existing local LeRobot modifications support explicit `init_state_offset` and
  reset IDs. This runner assumes those APIs; it is not a turnkey clean-install
  package. Fail rather than silently replacing explicit states with random resets.
- Official model code is imported externally. Loss-only perceptual modules are
  disabled during inference; missing inference weights are rejected.
- Follow upstream licenses, including **CC BY-NC-SA 4.0** for LVSM/weights.
  No upstream source tree, model weights, datasets or credentials are vendored.

## Run in the compatible environment

Download the pinned official source and weights, then set `ANYCAM_SOURCE` to
the source directory containing `LVSM/` and `openpi/`. Place weights under
`LVSM/ckpt/LIBERO-Plus_custom/`. Use an isolated environment; do not downgrade
the working parent environment to install the upstream requirements wholesale.

```bash
export ANYCAM_SOURCE=/path/to/AnyCamVLA
python experiments/anycam/evaluate.py --parent /path/to/pretrained_model --output /path/to/new-smoke --medium --nvs --smoke
python experiments/anycam/evaluate.py --parent /path/to/pretrained_model --output /path/to/new-lvsm --medium --nvs
python experiments/anycam/evaluate.py --parent /path/to/pretrained_model --output /path/to/new-baseline --medium
```

Inspect the smoke image before the full pilot. Output directories must be new.
The published runner removes the old RL-module import and makes machine paths
configurable; inference logic is retained. Packaging checks do not constitute
a new evaluation of the published refactor. No new evaluation is auto-started.
