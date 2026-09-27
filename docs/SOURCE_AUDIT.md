# Source and configuration audit — 2026-09-28

Scope: actual cloud installation used for recent evaluations, not a local Git checkout (workspace root is not a Git repository). Read-only inspection of runtime and weights; no evaluation rerun or dependency changes.

## Findings

- LeRobot0.6.1: verified490 Python files against installed distribution RECORD hashes. Exactly3 differ: envs/configs.py, envs/libero.py, scripts/lerobot_eval.py. SmolVLA policy/processor/trainer sources match installed RECORD. RECORD is installation provenance, not independent proof of equivalence to the paper's historical commit.
- hf_libero0.1.4:79 Python files and robosuite1.4.0:159 files match their installed RECORD hashes.
- Diffs against preserved pre-edit backups: initial-state offset parameter and its forwarding; recording feature/schema/image/state conversion changes. No success-rule modifications in those diffs. Offset0 used by formal evaluations has unchanged initialization arithmetic. Formal eval used recording=false, so recording-only changes do not explain the reported scores. Runtime initial-state sequence has not been independently traced in this audit.
- MuJoCo3.8.1 is installed. Upstream issue4390 reports task5 spatial reset physics differences from3.4.0 onward, including3.8.1. This is an external reproduction report, not yet locally isolated. It can undermine comparisons with other environments and may affect policies differently. https://github.com/huggingface/lerobot/issues/4390
- Parent saved training config:100000steps,batch64,lr1e-4,warmup1000,scheduler decay30000, frozen vision/expert-only, prefix_length=-1. Training100k does NOT mean cosine decay stretched to100k. Paper's exact comparable scheduler still requires verification; don't call30000 an implementation bug or claim remaining steps do nothing.
- Execution horizon50 in parent saved config versus paper1; current full400 explicitly overrides horizon1,num_steps10 for both. Earlier h10/h50 scores not paper-matched. Current totals parent305/400 teacher31d285/400 remain observed scores under this runtime.
- Downloaded31d teacher is a different training recipe; cannot identify it as paper450M100k result. Historical child v2 normalization override is confirmed negative result but does not alter original parent checkpoint.

## Next diagnostic, not performed

Preserve current environment. Use an isolated compatible MuJoCo environment for a small fixed-state Spatial5 reset/rollout comparison; trace actual reset state IDs and verify state-file/BDDL asset hashes. Obtain paper-era code/dependency pins before claiming exact official reproduction. Do not replace dependencies or rerun400 episodes without a new implementation request.
