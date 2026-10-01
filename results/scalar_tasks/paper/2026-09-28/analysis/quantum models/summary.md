# QLSTM vs Q-sLSTM vs Q-sLSTM-log vs Q-sLSTM-sqrt vs Q-sLSTM (fk) vs QLSTM (fk) — scalar tasks

Sweep: `results/scalar_tasks/paper/2026-09-28`

Paired seeds with all 6 models on all six tasks: 20. Values are mean ± sd over seeds on the test split, best-validation checkpoint. RMSE per seed = sqrt(test MSE).

The per-epoch curves use validation MSE: the runs log train loss and validation MSE each epoch but evaluate the test split only once, at the best checkpoint.

## Test MSE

| task | QLSTM | Q-sLSTM | Q-sLSTM-log | Q-sLSTM-sqrt | Q-sLSTM (fk) | QLSTM (fk) | best |
|---|---|---|---|---|---|---|---|
| delay | 0.249 ± 0.0054 | 0.25 ± 0.011 | 0.249 ± 0.016 | 0.259 ± 0.011 | 0.00197 ± 0.00066 | 0.00185 ± 0.00055 | QLSTM (fk) |
| ema | 0.000682 ± 0.00033 | 0.004 ± 0.0016 | 0.00336 ± 0.00072 | 0.00389 ± 0.0018 | 3.18e-05 ± 1.2e-05 | 2.43e-05 ± 1.8e-05 | QLSTM (fk) |
| running_max | 0.018 ± 0.0024 | 0.00298 ± 0.00084 | 0.00398 ± 0.00039 | 0.00527 ± 0.0012 | 0.000447 ± 0.00032 | 0.000769 ± 0.00033 | Q-sLSTM (fk) |
| flip_flop | 0.131 ± 0.013 | 0.105 ± 0.014 | 0.141 ± 0.059 | 0.148 ± 0.036 | 4.69e-05 ± 3.3e-05 | 8.12e-05 ± 6.4e-05 | Q-sLSTM (fk) |
| narma | 0.00692 ± 0.00042 | 0.00572 ± 0.00044 | 0.00566 ± 0.00037 | 0.00561 ± 0.00032 | 0.00452 ± 0.00068 | 0.00435 ± 0.00064 | QLSTM (fk) |
| sine_next | 0.0137 ± 0.001 | 0.00752 ± 0.0015 | 0.00856 ± 0.0037 | 0.00904 ± 0.0013 | 0.000115 ± 5.3e-05 | 0.000146 ± 7.2e-05 | Q-sLSTM (fk) |

## Test MAE

| task | QLSTM | Q-sLSTM | Q-sLSTM-log | Q-sLSTM-sqrt | Q-sLSTM (fk) | QLSTM (fk) | best |
|---|---|---|---|---|---|---|---|
| delay | 0.415 ± 0.0064 | 0.42 ± 0.012 | 0.415 ± 0.017 | 0.428 ± 0.011 | 0.0342 ± 0.0059 | 0.033 ± 0.0056 | QLSTM (fk) |
| ema | 0.019 ± 0.0051 | 0.0455 ± 0.0086 | 0.0415 ± 0.0043 | 0.0443 ± 0.0092 | 0.00344 ± 0.00068 | 0.00321 ± 0.00099 | QLSTM (fk) |
| running_max | 0.0956 ± 0.0058 | 0.042 ± 0.0072 | 0.0495 ± 0.0024 | 0.0565 ± 0.0064 | 0.0143 ± 0.0026 | 0.0203 ± 0.004 | Q-sLSTM (fk) |
| flip_flop | 0.244 ± 0.015 | 0.222 ± 0.022 | 0.254 ± 0.085 | 0.277 ± 0.05 | 0.00374 ± 0.0014 | 0.00567 ± 0.0021 | Q-sLSTM (fk) |
| narma | 0.0651 ± 0.002 | 0.0592 ± 0.0026 | 0.0587 ± 0.0021 | 0.0585 ± 0.0018 | 0.0518 ± 0.0044 | 0.0507 ± 0.004 | QLSTM (fk) |
| sine_next | 0.0904 ± 0.0033 | 0.0659 ± 0.0073 | 0.0699 ± 0.013 | 0.0721 ± 0.0058 | 0.00757 ± 0.0017 | 0.00852 ± 0.0025 | Q-sLSTM (fk) |

## Test RMSE

| task | QLSTM | Q-sLSTM | Q-sLSTM-log | Q-sLSTM-sqrt | Q-sLSTM (fk) | QLSTM (fk) | best |
|---|---|---|---|---|---|---|---|
| delay | 0.499 ± 0.0054 | 0.5 ± 0.011 | 0.499 ± 0.016 | 0.508 ± 0.011 | 0.0439 ± 0.0073 | 0.0424 ± 0.0071 | QLSTM (fk) |
| ema | 0.0254 ± 0.006 | 0.0621 ± 0.012 | 0.0576 ± 0.0067 | 0.0609 ± 0.014 | 0.00553 ± 0.0011 | 0.00474 ± 0.0014 | QLSTM (fk) |
| running_max | 0.134 ± 0.0085 | 0.054 ± 0.0083 | 0.063 ± 0.0031 | 0.0722 ± 0.0081 | 0.0203 ± 0.0061 | 0.0271 ± 0.0059 | Q-sLSTM (fk) |
| flip_flop | 0.361 ± 0.017 | 0.324 ± 0.022 | 0.362 ± 0.1 | 0.38 ± 0.057 | 0.00648 ± 0.0023 | 0.00849 ± 0.0031 | Q-sLSTM (fk) |
| narma | 0.0832 ± 0.0026 | 0.0756 ± 0.0029 | 0.0752 ± 0.0024 | 0.0749 ± 0.0021 | 0.067 ± 0.0052 | 0.0658 ± 0.0049 | QLSTM (fk) |
| sine_next | 0.117 ± 0.0043 | 0.0863 ± 0.0087 | 0.0911 ± 0.016 | 0.0949 ± 0.0066 | 0.0105 ± 0.0023 | 0.0117 ± 0.003 | Q-sLSTM (fk) |

## Best validation MSE and epoch

| task | QLSTM | Q-sLSTM | Q-sLSTM-log | Q-sLSTM-sqrt | Q-sLSTM (fk) | QLSTM (fk) |
|---|---|---|---|---|---|---|
| delay | 0.251 ± 0.0075 (ep 60 ± 1) | 0.252 ± 0.012 (ep 59 ± 1) | 0.251 ± 0.018 (ep 59 ± 1) | 0.261 ± 0.011 (ep 60 ± 1) | 0.00198 ± 0.00064 (ep 58 ± 2) | 0.00186 ± 0.00056 (ep 58 ± 3) |
| ema | 0.000669 ± 0.00035 (ep 60 ± 1) | 0.00395 ± 0.0016 (ep 60 ± 1) | 0.00327 ± 0.00066 (ep 60 ± 1) | 0.0038 ± 0.0017 (ep 60 ± 1) | 3.29e-05 ± 1.9e-05 (ep 56 ± 3) | 2.39e-05 ± 1.8e-05 (ep 57 ± 3) |
| running_max | 0.0175 ± 0.0017 (ep 59 ± 1) | 0.003 ± 0.0009 (ep 59 ± 1) | 0.00399 ± 0.00058 (ep 59 ± 1) | 0.00527 ± 0.0015 (ep 59 ± 1) | 0.000443 ± 0.00032 (ep 59 ± 1) | 0.000782 ± 0.00037 (ep 59 ± 1) |
| flip_flop | 0.13 ± 0.012 (ep 57 ± 4) | 0.103 ± 0.015 (ep 58 ± 2) | 0.138 ± 0.058 (ep 57 ± 3) | 0.147 ± 0.035 (ep 57 ± 2) | 4.45e-05 ± 3.5e-05 (ep 60 ± 1) | 7.99e-05 ± 6.1e-05 (ep 58 ± 3) |
| narma | 0.00703 ± 0.00067 (ep 59 ± 2) | 0.00587 ± 0.0007 (ep 59 ± 2) | 0.00579 ± 0.00064 (ep 58 ± 3) | 0.00576 ± 0.00067 (ep 58 ± 2) | 0.00464 ± 0.00087 (ep 56 ± 4) | 0.00445 ± 0.00086 (ep 56 ± 6) |
| sine_next | 0.0135 ± 0.0013 (ep 59 ± 1) | 0.0074 ± 0.0016 (ep 60 ± 1) | 0.00849 ± 0.0038 (ep 59 ± 1) | 0.00894 ± 0.0014 (ep 59 ± 1) | 0.000114 ± 5.1e-05 (ep 59 ± 1) | 0.000148 ± 7.5e-05 (ep 60 ± 1) |

## Paired test-MSE differences (a − b; negative = a lower)

Wilcoxon signed-rank over seeds; `wins` = seeds where a has lower MSE.

| task | a − b | mean diff ± sd | wins | p |
|---|---|---|---|---|
| delay | QLSTM − Q-sLSTM | -6.78e-04 ± 1.16e-02 | 10/20 | 0.841 |
| delay | QLSTM − Q-sLSTM-log | +1.16e-04 ± 1.49e-02 | 9/20 | 0.674 |
| delay | QLSTM − Q-sLSTM-sqrt | -9.46e-03 ± 1.15e-02 | 18/20 | 0.000168 |
| delay | QLSTM − Q-sLSTM (fk) | +2.47e-01 ± 5.32e-03 | 0/20 | 1.91e-06 |
| delay | QLSTM − QLSTM (fk) | +2.47e-01 ± 5.20e-03 | 0/20 | 1.91e-06 |
| delay | Q-sLSTM − Q-sLSTM-log | +7.94e-04 ± 1.56e-02 | 9/20 | 0.898 |
| delay | Q-sLSTM − Q-sLSTM-sqrt | -8.78e-03 ± 1.07e-02 | 18/20 | 0.000586 |
| delay | Q-sLSTM − Q-sLSTM (fk) | +2.48e-01 ± 1.09e-02 | 0/20 | 1.91e-06 |
| delay | Q-sLSTM − QLSTM (fk) | +2.48e-01 ± 1.09e-02 | 0/20 | 1.91e-06 |
| delay | Q-sLSTM-log − Q-sLSTM-sqrt | -9.57e-03 ± 2.09e-02 | 14/20 | 0.0362 |
| delay | Q-sLSTM-log − Q-sLSTM (fk) | +2.47e-01 ± 1.60e-02 | 0/20 | 1.91e-06 |
| delay | Q-sLSTM-log − QLSTM (fk) | +2.47e-01 ± 1.61e-02 | 0/20 | 1.91e-06 |
| delay | Q-sLSTM-sqrt − Q-sLSTM (fk) | +2.57e-01 ± 1.12e-02 | 0/20 | 1.91e-06 |
| delay | Q-sLSTM-sqrt − QLSTM (fk) | +2.57e-01 ± 1.10e-02 | 0/20 | 1.91e-06 |
| delay | Q-sLSTM (fk) − QLSTM (fk) | +1.26e-04 ± 8.63e-04 | 9/20 | 0.648 |
| ema | QLSTM − Q-sLSTM | -3.31e-03 ± 1.81e-03 | 20/20 | 1.91e-06 |
| ema | QLSTM − Q-sLSTM-log | -2.68e-03 ± 8.55e-04 | 20/20 | 1.91e-06 |
| ema | QLSTM − Q-sLSTM-sqrt | -3.21e-03 ± 1.94e-03 | 19/20 | 3.81e-06 |
| ema | QLSTM − Q-sLSTM (fk) | +6.50e-04 ± 3.36e-04 | 0/20 | 1.91e-06 |
| ema | QLSTM − QLSTM (fk) | +6.58e-04 ± 3.32e-04 | 0/20 | 1.91e-06 |
| ema | Q-sLSTM − Q-sLSTM-log | +6.37e-04 ± 1.56e-03 | 8/20 | 0.165 |
| ema | Q-sLSTM − Q-sLSTM-sqrt | +1.08e-04 ± 7.09e-04 | 6/20 | 0.189 |
| ema | Q-sLSTM − Q-sLSTM (fk) | +3.96e-03 ± 1.63e-03 | 0/20 | 1.91e-06 |
| ema | Q-sLSTM − QLSTM (fk) | +3.97e-03 ± 1.64e-03 | 0/20 | 1.91e-06 |
| ema | Q-sLSTM-log − Q-sLSTM-sqrt | -5.29e-04 ± 1.66e-03 | 12/20 | 0.475 |
| ema | Q-sLSTM-log − Q-sLSTM (fk) | +3.33e-03 ± 7.13e-04 | 0/20 | 1.91e-06 |
| ema | Q-sLSTM-log − QLSTM (fk) | +3.33e-03 ± 7.19e-04 | 0/20 | 1.91e-06 |
| ema | Q-sLSTM-sqrt − Q-sLSTM (fk) | +3.86e-03 ± 1.77e-03 | 0/20 | 1.91e-06 |
| ema | Q-sLSTM-sqrt − QLSTM (fk) | +3.86e-03 ± 1.78e-03 | 0/20 | 1.91e-06 |
| ema | Q-sLSTM (fk) − QLSTM (fk) | +7.48e-06 ± 2.06e-05 | 3/20 | 0.00143 |
| running_max | QLSTM − Q-sLSTM | +1.50e-02 ± 2.80e-03 | 0/20 | 1.91e-06 |
| running_max | QLSTM − Q-sLSTM-log | +1.40e-02 ± 2.56e-03 | 0/20 | 1.91e-06 |
| running_max | QLSTM − Q-sLSTM-sqrt | +1.27e-02 ± 2.96e-03 | 0/20 | 1.91e-06 |
| running_max | QLSTM − Q-sLSTM (fk) | +1.75e-02 ± 2.39e-03 | 0/20 | 1.91e-06 |
| running_max | QLSTM − QLSTM (fk) | +1.72e-02 ± 2.38e-03 | 0/20 | 1.91e-06 |
| running_max | Q-sLSTM − Q-sLSTM-log | -9.98e-04 ± 6.92e-04 | 19/20 | 3.81e-06 |
| running_max | Q-sLSTM − Q-sLSTM-sqrt | -2.29e-03 ± 1.22e-03 | 20/20 | 1.91e-06 |
| running_max | Q-sLSTM − Q-sLSTM (fk) | +2.53e-03 ± 9.16e-04 | 0/20 | 1.91e-06 |
| running_max | Q-sLSTM − QLSTM (fk) | +2.21e-03 ± 8.70e-04 | 0/20 | 1.91e-06 |
| running_max | Q-sLSTM-log − Q-sLSTM-sqrt | -1.29e-03 ± 1.18e-03 | 20/20 | 1.91e-06 |
| running_max | Q-sLSTM-log − Q-sLSTM (fk) | +3.53e-03 ± 5.80e-04 | 0/20 | 1.91e-06 |
| running_max | Q-sLSTM-log − QLSTM (fk) | +3.21e-03 ± 4.35e-04 | 0/20 | 1.91e-06 |
| running_max | Q-sLSTM-sqrt − Q-sLSTM (fk) | +4.82e-03 ± 1.27e-03 | 0/20 | 1.91e-06 |
| running_max | Q-sLSTM-sqrt − QLSTM (fk) | +4.50e-03 ± 1.28e-03 | 0/20 | 1.91e-06 |
| running_max | Q-sLSTM (fk) − QLSTM (fk) | -3.22e-04 ± 4.56e-04 | 16/20 | 0.00558 |
| flip_flop | QLSTM − Q-sLSTM | +2.55e-02 ± 1.78e-02 | 1/20 | 3.81e-06 |
| flip_flop | QLSTM − Q-sLSTM-log | -1.00e-02 ± 5.46e-02 | 15/20 | 0.261 |
| flip_flop | QLSTM − Q-sLSTM-sqrt | -1.67e-02 ± 3.38e-02 | 16/20 | 0.0172 |
| flip_flop | QLSTM − Q-sLSTM (fk) | +1.31e-01 ± 1.27e-02 | 0/20 | 1.91e-06 |
| flip_flop | QLSTM − QLSTM (fk) | +1.31e-01 ± 1.27e-02 | 0/20 | 1.91e-06 |
| flip_flop | Q-sLSTM − Q-sLSTM-log | -3.55e-02 ± 5.62e-02 | 15/20 | 0.0296 |
| flip_flop | Q-sLSTM − Q-sLSTM-sqrt | -4.21e-02 ± 3.87e-02 | 18/20 | 0.00102 |
| flip_flop | Q-sLSTM − Q-sLSTM (fk) | +1.05e-01 ± 1.37e-02 | 0/20 | 1.91e-06 |
| flip_flop | Q-sLSTM − QLSTM (fk) | +1.05e-01 ± 1.37e-02 | 0/20 | 1.91e-06 |
| flip_flop | Q-sLSTM-log − Q-sLSTM-sqrt | -6.65e-03 ± 3.58e-02 | 7/20 | 0.701 |
| flip_flop | Q-sLSTM-log − Q-sLSTM (fk) | +1.41e-01 ± 5.91e-02 | 0/20 | 1.91e-06 |
| flip_flop | Q-sLSTM-log − QLSTM (fk) | +1.41e-01 ± 5.91e-02 | 0/20 | 1.91e-06 |
| flip_flop | Q-sLSTM-sqrt − Q-sLSTM (fk) | +1.47e-01 ± 3.57e-02 | 0/20 | 1.91e-06 |
| flip_flop | Q-sLSTM-sqrt − QLSTM (fk) | +1.47e-01 ± 3.57e-02 | 0/20 | 1.91e-06 |
| flip_flop | Q-sLSTM (fk) − QLSTM (fk) | -3.43e-05 ± 6.51e-05 | 14/20 | 0.0153 |
| narma | QLSTM − Q-sLSTM | +1.20e-03 ± 5.54e-04 | 0/20 | 1.91e-06 |
| narma | QLSTM − Q-sLSTM-log | +1.26e-03 ± 5.78e-04 | 1/20 | 3.81e-06 |
| narma | QLSTM − Q-sLSTM-sqrt | +1.31e-03 ± 5.32e-04 | 1/20 | 3.81e-06 |
| narma | QLSTM − Q-sLSTM (fk) | +2.40e-03 ± 8.23e-04 | 0/20 | 1.91e-06 |
| narma | QLSTM − QLSTM (fk) | +2.57e-03 ± 5.81e-04 | 0/20 | 1.91e-06 |
| narma | Q-sLSTM − Q-sLSTM-log | +6.07e-05 ± 2.95e-04 | 10/20 | 0.43 |
| narma | Q-sLSTM − Q-sLSTM-sqrt | +1.16e-04 ± 2.47e-04 | 3/20 | 0.0172 |
| narma | Q-sLSTM − Q-sLSTM (fk) | +1.20e-03 ± 6.77e-04 | 0/20 | 1.91e-06 |
| narma | Q-sLSTM − QLSTM (fk) | +1.38e-03 ± 5.14e-04 | 0/20 | 1.91e-06 |
| narma | Q-sLSTM-log − Q-sLSTM-sqrt | +5.54e-05 ± 2.52e-04 | 9/20 | 0.701 |
| narma | Q-sLSTM-log − Q-sLSTM (fk) | +1.14e-03 ± 5.99e-04 | 0/20 | 1.91e-06 |
| narma | Q-sLSTM-log − QLSTM (fk) | +1.32e-03 ± 6.04e-04 | 0/20 | 1.91e-06 |
| narma | Q-sLSTM-sqrt − Q-sLSTM (fk) | +1.09e-03 ± 6.41e-04 | 0/20 | 1.91e-06 |
| narma | Q-sLSTM-sqrt − QLSTM (fk) | +1.26e-03 ± 5.99e-04 | 0/20 | 1.91e-06 |
| narma | Q-sLSTM (fk) − QLSTM (fk) | +1.73e-04 ± 1.02e-03 | 8/20 | 0.452 |
| sine_next | QLSTM − Q-sLSTM | +6.17e-03 ± 1.69e-03 | 0/20 | 1.91e-06 |
| sine_next | QLSTM − Q-sLSTM-log | +5.14e-03 ± 4.14e-03 | 2/20 | 0.000851 |
| sine_next | QLSTM − Q-sLSTM-sqrt | +4.65e-03 ± 1.71e-03 | 0/20 | 1.91e-06 |
| sine_next | QLSTM − Q-sLSTM (fk) | +1.36e-02 ± 1.03e-03 | 0/20 | 1.91e-06 |
| sine_next | QLSTM − QLSTM (fk) | +1.35e-02 ± 1.02e-03 | 0/20 | 1.91e-06 |
| sine_next | Q-sLSTM − Q-sLSTM-log | -1.03e-03 ± 3.18e-03 | 11/20 | 0.245 |
| sine_next | Q-sLSTM − Q-sLSTM-sqrt | -1.52e-03 ± 9.09e-04 | 19/20 | 9.54e-06 |
| sine_next | Q-sLSTM − Q-sLSTM (fk) | +7.41e-03 ± 1.51e-03 | 0/20 | 1.91e-06 |
| sine_next | Q-sLSTM − QLSTM (fk) | +7.38e-03 ± 1.50e-03 | 0/20 | 1.91e-06 |
| sine_next | Q-sLSTM-log − Q-sLSTM-sqrt | -4.85e-04 ± 3.20e-03 | 16/20 | 0.0897 |
| sine_next | Q-sLSTM-log − Q-sLSTM (fk) | +8.44e-03 ± 3.67e-03 | 0/20 | 1.91e-06 |
| sine_next | Q-sLSTM-log − QLSTM (fk) | +8.41e-03 ± 3.64e-03 | 0/20 | 1.91e-06 |
| sine_next | Q-sLSTM-sqrt − Q-sLSTM (fk) | +8.93e-03 ± 1.29e-03 | 0/20 | 1.91e-06 |
| sine_next | Q-sLSTM-sqrt − QLSTM (fk) | +8.90e-03 ± 1.29e-03 | 0/20 | 1.91e-06 |
| sine_next | Q-sLSTM (fk) − QLSTM (fk) | -3.09e-05 ± 8.82e-05 | 13/20 | 0.114 |

## Figures

Log y-axis; linear-axis copies of each are in `linear/`.

- `test_mse.png`
- `test_mae.png`
- `test_rmse.png`
- `mse_vs_timestep.png`
- `mae_vs_timestep.png`
- `rmse_vs_timestep.png`
- `loss_vs_epoch.png`
- `train_loss_vs_epoch.png`
- `val_loss_vs_epoch.png`
