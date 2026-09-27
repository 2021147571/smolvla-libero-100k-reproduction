# Full400 horizon1 paired evaluation

Both: execution horizon1, flow-matching steps10, seed1000, AMPfalse, batch1, init states0-9. Each model keeps its native processors. Teacher revision31d453f7.

| Suite | Parent | Teacher | Teacher minus parent |
|---|---:|---:|---:|
| spatial | 87/100 | 78/100 | -9 |
| object | 83/100 | 74/100 | -9 |
| goal | 82/100 | 76/100 | -6 |
| 10 | 53/100 | 57/100 | +4 |
| Total | 305/400 (76.25%) | 285/400 (71.25%) | -20 |

## Task-level results

| Suite/task | Parent /10 | Teacher /10 | Teacher-only | Parent-only |
|---|---:|---:|---:|---:|
| spatial/0 | 9 | 10 | 1 | 0 |
| spatial/1 | 9 | 10 | 1 | 0 |
| spatial/2 | 9 | 10 | 1 | 0 |
| spatial/3 | 9 | 9 | 1 | 1 |
| spatial/4 | 9 | 6 | 0 | 3 |
| spatial/5 | 6 | 0 | 0 | 6 |
| spatial/6 | 10 | 10 | 0 | 0 |
| spatial/7 | 9 | 8 | 1 | 2 |
| spatial/8 | 9 | 7 | 0 | 2 |
| spatial/9 | 8 | 8 | 0 | 0 |
| object/0 | 9 | 7 | 1 | 3 |
| object/1 | 10 | 5 | 0 | 5 |
| object/2 | 10 | 9 | 0 | 1 |
| object/3 | 9 | 8 | 1 | 2 |
| object/4 | 5 | 5 | 2 | 2 |
| object/5 | 5 | 5 | 1 | 1 |
| object/6 | 10 | 8 | 0 | 2 |
| object/7 | 10 | 10 | 0 | 0 |
| object/8 | 7 | 9 | 3 | 1 |
| object/9 | 8 | 8 | 2 | 2 |
| goal/0 | 10 | 6 | 0 | 4 |
| goal/1 | 10 | 9 | 0 | 1 |
| goal/2 | 9 | 9 | 1 | 1 |
| goal/3 | 6 | 6 | 3 | 3 |
| goal/4 | 10 | 10 | 0 | 0 |
| goal/5 | 10 | 10 | 0 | 0 |
| goal/6 | 5 | 3 | 1 | 3 |
| goal/7 | 9 | 9 | 1 | 1 |
| goal/8 | 9 | 10 | 1 | 0 |
| goal/9 | 4 | 4 | 3 | 3 |
| 10/0 | 3 | 0 | 0 | 3 |
| 10/1 | 6 | 6 | 2 | 2 |
| 10/2 | 9 | 8 | 0 | 1 |
| 10/3 | 10 | 10 | 0 | 0 |
| 10/4 | 2 | 4 | 3 | 1 |
| 10/5 | 6 | 9 | 3 | 0 |
| 10/6 | 4 | 6 | 4 | 2 |
| 10/7 | 3 | 5 | 3 | 1 |
| 10/8 | 2 | 2 | 2 | 2 |
| 10/9 | 8 | 7 | 1 | 2 |

Episode-index pairing uses corresponding initial states, but sequential stochastic RNG may diverge. These are observed results under this implementation, not proof of exact paper reproduction. Test states have already been inspected, not an untouched future confirmation set.

Completed22:41:43 +08:00 on2026-09-27. No finetuning or correction collection authorized now. Keep cloud and local computer ON; wait for user.
