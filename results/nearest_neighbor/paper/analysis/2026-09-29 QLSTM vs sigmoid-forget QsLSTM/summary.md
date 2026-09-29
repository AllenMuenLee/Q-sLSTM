# QLSTM vs Q-sLSTM vs Q-sLSTM-log

Sweeps: `results/nearest_neighbor/paper/2026-09-21 QLSTM vs QsLSTM/seed_1023290587/qlstm`, `results/nearest_neighbor/paper/2026-09-21 QLSTM vs QsLSTM/seed_1106723791/qlstm`, `results/nearest_neighbor/paper/2026-09-21 QLSTM vs QsLSTM/seed_1148376891/qlstm`, `results/nearest_neighbor/paper/2026-09-21 QLSTM vs QsLSTM/seed_1211904294/qlstm`, `results/nearest_neighbor/paper/2026-09-21 QLSTM vs QsLSTM/seed_1212609727/qlstm`, `results/nearest_neighbor/paper/2026-09-21 QLSTM vs QsLSTM/seed_1668868548/qlstm`, `results/nearest_neighbor/paper/2026-09-21 QLSTM vs QsLSTM/seed_1710807515/qlstm`, `results/nearest_neighbor/paper/2026-09-21 QLSTM vs QsLSTM/seed_1716202887/qlstm`, `results/nearest_neighbor/paper/2026-09-21 QLSTM vs QsLSTM/seed_1806016308/qlstm`, `results/nearest_neighbor/paper/2026-09-21 QLSTM vs QsLSTM/seed_1885117419/qlstm`, `results/nearest_neighbor/paper/2026-09-21 QLSTM vs QsLSTM/seed_1952804984/qlstm`, `results/nearest_neighbor/paper/2026-09-21 QLSTM vs QsLSTM/seed_1992756749/qlstm`, `results/nearest_neighbor/paper/2026-09-21 QLSTM vs QsLSTM/seed_2036585866/qlstm`, `results/nearest_neighbor/paper/2026-09-21 QLSTM vs QsLSTM/seed_266064982/qlstm`, `results/nearest_neighbor/paper/2026-09-21 QLSTM vs QsLSTM/seed_435874469/qlstm`, `results/nearest_neighbor/paper/2026-09-21 QLSTM vs QsLSTM/seed_484324222/qlstm`, `results/nearest_neighbor/paper/2026-09-21 QLSTM vs QsLSTM/seed_83453494/qlstm`, `results/nearest_neighbor/paper/2026-09-21 QLSTM vs QsLSTM/seed_867702092/qlstm`, `results/nearest_neighbor/paper/2026-09-21 QLSTM vs QsLSTM/seed_88159254/qlstm`, `results/nearest_neighbor/paper/2026-09-21 QLSTM vs QsLSTM/seed_975428833/qlstm`, `results/nearest_neighbor/paper/2026-09-27`

Paired seeds with all three models: 19. Differences are a − b over seeds; negative = first model lower. CI: Student-t over seed-level differences; p: Wilcoxon signed-rank.

Recurrences: qlstm = `n/a`; qslstm = `xlstm_stabilized_sigmoid_forget_v1`; qslstm_log = `log_input_sigmoid_forget_binary_scale_v1`

## Validation (best epoch)

| model | best val MSE | best epoch |
|---|---|---|
| QLSTM | 0.07923 ± 0.00264 | 93.3 ± 10.1 |
| Q-sLSTM | 0.07859 ± 0.00268 | 92.0 ± 9.4 |
| Q-sLSTM-log | 0.07876 ± 0.00295 | 89.7 ± 11.8 |

## held-out (trained length)

### Overview by case type (mean ± sd over seeds)

| model | case | revision gain | best epoch | MSE | MAE | RMSE |
|---|---|---|---|---|---|---|
| QLSTM | all | 0.04067 ± 0.00330 | 93.2 ± 10.4 | 0.11106 ± 0.00288 | 0.29772 ± 0.00487 | 0.33323 ± 0.00430 |
| QLSTM | iid | 0.03176 ± 0.00246 | 93.2 ± 10.4 | 0.07883 ± 0.00277 | 0.24083 ± 0.00528 | 0.28073 ± 0.00497 |
| QLSTM | late | 0.03817 ± 0.00376 | 93.2 ± 10.4 | 0.10379 ± 0.00277 | 0.29034 ± 0.00497 | 0.32214 ± 0.00427 |
| QLSTM | early | 0.04758 ± 0.00605 | 93.2 ± 10.4 | 0.10766 ± 0.00600 | 0.29496 ± 0.01095 | 0.32799 ± 0.00918 |
| QLSTM | near_best | 0.04802 ± 0.00561 | 93.2 ± 10.4 | 0.15396 ± 0.00712 | 0.36474 ± 0.01003 | 0.39228 ± 0.00908 |
| Q-sLSTM | all | 0.04858 ± 0.00381 | 93.3 ± 7.5 | 0.11005 ± 0.00299 | 0.29587 ± 0.00512 | 0.33172 ± 0.00450 |
| Q-sLSTM | iid | 0.03571 ± 0.00253 | 93.3 ± 7.5 | 0.07796 ± 0.00260 | 0.23880 ± 0.00512 | 0.27918 ± 0.00469 |
| Q-sLSTM | late | 0.03972 ± 0.00403 | 93.3 ± 7.5 | 0.10354 ± 0.00307 | 0.28966 ± 0.00538 | 0.32175 ± 0.00475 |
| Q-sLSTM | early | 0.06184 ± 0.00704 | 93.3 ± 7.5 | 0.10696 ± 0.00615 | 0.29364 ± 0.01095 | 0.32692 ± 0.00942 |
| Q-sLSTM | near_best | 0.06286 ± 0.00898 | 93.3 ± 7.5 | 0.15175 ± 0.00579 | 0.36137 ± 0.00813 | 0.38949 ± 0.00742 |
| Q-sLSTM-log | all | 0.04892 ± 0.00385 | 89.7 ± 11.8 | 0.11030 ± 0.00284 | 0.29628 ± 0.00483 | 0.33208 ± 0.00425 |
| Q-sLSTM-log | iid | 0.03608 ± 0.00273 | 89.7 ± 11.8 | 0.07809 ± 0.00261 | 0.23918 ± 0.00494 | 0.27941 ± 0.00470 |
| Q-sLSTM-log | late | 0.04106 ± 0.00453 | 89.7 ± 11.8 | 0.10356 ± 0.00289 | 0.28966 ± 0.00504 | 0.32178 ± 0.00447 |
| Q-sLSTM-log | early | 0.06195 ± 0.00662 | 89.7 ± 11.8 | 0.10694 ± 0.00605 | 0.29364 ± 0.01099 | 0.32689 ± 0.00928 |
| Q-sLSTM-log | near_best | 0.06203 ± 0.00756 | 89.7 ± 11.8 | 0.15260 ± 0.00583 | 0.36265 ± 0.00813 | 0.39057 ± 0.00745 |

### Mean ± sd over seeds (all cases)

| metric | QLSTM | Q-sLSTM | Q-sLSTM-log |
|---|---|---|---|
| MSE | 0.11106 ± 0.00288 | 0.11005 ± 0.00299 | 0.11030 ± 0.00284 |
| MAE | 0.29772 ± 0.00487 | 0.29587 ± 0.00512 | 0.29628 ± 0.00483 |
| final-step MSE | 0.11899 ± 0.00334 | 0.11830 ± 0.00373 | 0.11855 ± 0.00297 |
| event-step MSE | 0.09013 ± 0.00308 | 0.08931 ± 0.00456 | 0.08853 ± 0.00440 |
| non-event MSE | 0.11269 ± 0.00292 | 0.11165 ± 0.00307 | 0.11198 ± 0.00289 |
| non-event drift | 0.03084 ± 0.00227 | 0.02952 ± 0.00367 | 0.03062 ± 0.00362 |
| revision gain (higher = better) | 0.04067 ± 0.00330 | 0.04858 ± 0.00381 | 0.04892 ± 0.00385 |
| alpha contrast (event - non-event) | 0.03294 ± 0.00642 | 0.03901 ± 0.01253 | 0.03677 ± 0.01251 |

### MSE by case type

| case | QLSTM | Q-sLSTM | Q-sLSTM-log |
|---|---|---|---|
| all | 0.11106 | 0.11005 | 0.11030 |
| iid | 0.07883 | 0.07796 | 0.07809 |
| late | 0.10379 | 0.10354 | 0.10356 |
| early | 0.10766 | 0.10696 | 0.10694 |
| near_best | 0.15396 | 0.15175 | 0.15260 |

### Paired differences

| metric | case | Q-sLSTM − QLSTM | Q-sLSTM-log − QLSTM | Q-sLSTM-log − Q-sLSTM |
|---|---|---|---|---|
| MSE | all | -0.00101 (-0.9%) [-0.00154, -0.00047] 16/19, p=6.4e-04 | -0.00076 (-0.7%) [-0.00127, -0.00025] 15/19, p=0.003 | +0.00024 (+0.2%) [-0.00033, +0.00081] 8/19, p=0.441 |
| MAE | all | -0.00185 (-0.6%) [-0.00250, -0.00120] 18/19, p=1.1e-05 | -0.00144 (-0.5%) [-0.00201, -0.00086] 18/19, p=3.8e-05 | +0.00041 (+0.1%) [-0.00027, +0.00109] 7/19, p=0.182 |
| final-step MSE | all | -0.00069 (-0.6%) [-0.00168, +0.00030] 11/19, p=0.145 | -0.00043 (-0.4%) [-0.00133, +0.00047] 13/19, p=0.210 | +0.00026 (+0.2%) [-0.00063, +0.00115] 8/19, p=0.515 |
| event-step MSE | all | -0.00082 (-0.9%) [-0.00304, +0.00140] 11/19, p=0.395 | -0.00160 (-1.8%) [-0.00341, +0.00022] 12/19, p=0.104 | -0.00078 (-0.9%) [-0.00273, +0.00117] 12/19, p=0.541 |
| non-event MSE | all | -0.00104 (-0.9%) [-0.00159, -0.00049] 16/19, p=5.2e-04 | -0.00071 (-0.6%) [-0.00122, -0.00019] 15/19, p=0.008 | +0.00033 (+0.3%) [-0.00028, +0.00094] 5/19, p=0.156 |
| non-event drift | all | -0.00133 (-4.3%) [-0.00338, +0.00073] 12/19, p=0.123 | -0.00022 (-0.7%) [-0.00181, +0.00138] 9/19, p=0.768 | +0.00111 (+3.8%) [-0.00071, +0.00293] 7/19, p=0.169 |
| revision gain (higher = better) | all | +0.00791 (+19.4%) [+0.00536, +0.01046] 1/19, p=2.7e-05 | +0.00825 (+20.3%) [+0.00610, +0.01040] 1/19, p=7.6e-06 | +0.00034 (+0.7%) [-0.00183, +0.00251] 8/19, p=0.568 |
| alpha contrast (event - non-event) | all | +0.00607 (+18.4%) [-0.00079, +0.01293] 7/19, p=0.096 | +0.00383 (+11.6%) [-0.00231, +0.00997] 9/19, p=0.225 | -0.00224 (-5.7%) [-0.01065, +0.00617] 12/19, p=0.541 |
| MSE | iid | -0.00087 (-1.1%) [-0.00122, -0.00053] 18/19, p=5.3e-05 | -0.00074 (-0.9%) [-0.00106, -0.00042] 16/19, p=2.1e-04 | +0.00013 (+0.2%) [-0.00015, +0.00041] 8/19, p=0.352 |
| MSE | late | -0.00025 (-0.2%) [-0.00053, +0.00003] 13/19, p=0.134 | -0.00023 (-0.2%) [-0.00047, +0.00000] 13/19, p=0.055 | +0.00002 (+0.0%) [-0.00031, +0.00034] 7/19, p=0.953 |
| MSE | early | -0.00070 (-0.6%) [-0.00118, -0.00021] 13/19, p=0.011 | -0.00072 (-0.7%) [-0.00118, -0.00026] 15/19, p=0.009 | -0.00002 (-0.0%) [-0.00045, +0.00040] 10/19, p=0.829 |
| MSE | near_best | -0.00221 (-1.4%) [-0.00454, +0.00012] 13/19, p=0.066 | -0.00137 (-0.9%) [-0.00360, +0.00087] 12/19, p=0.312 | +0.00085 (+0.6%) [-0.00134, +0.00303] 7/19, p=0.395 |

