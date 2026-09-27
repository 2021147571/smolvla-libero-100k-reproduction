# Matched Short-300 h10 comparison

Protocol: horizon10, seed1000, num_steps10, AMPfalse, batch1, states0-9. Native processors retained for each model.

| Suite | Parent | Teacher | Teacher minus parent |
|---|---:|---:|---:|
| spatial | 84/100 | 75/100 | -9 |
| object | 95/100 | 88/100 | -7 |
| goal | 89/100 | 87/100 | -2 |
| Total | 268/300 | 250/300 | -18 |

Per-task counts follow. Episode-index pairing shares initial states, but sequential stochastic RNG streams can diverge; do not treat these as identically coupled noise trajectories.

spatial: task0 P8/T10, task1 P10/T9, task2 P9/T10, task3 P10/T9, task4 P9/T5, task5 P5/T0, task6 P8/T8, task7 P9/T8, task8 P8/T7, task9 P8/T9
object: task0 P10/T9, task1 P10/T10, task2 P10/T8, task3 P9/T9, task4 P9/T8, task5 P8/T7, task6 P10/T10, task7 P9/T8, task8 P10/T9, task9 P10/T10
goal: task0 P10/T10, task1 P10/T10, task2 P10/T8, task3 P8/T9, task4 P10/T10, task5 P9/T10, task6 P8/T4, task7 P10/T10, task8 P9/T9, task9 P5/T7
