# QLSTM vs Q-sLSTM vs Q-sLSTM (fk) — scalar tasks

Sweep: `results/scalar_tasks/paper/2026-09-28`

Paired seeds with all three models on all six tasks: 20. Values are mean ± sd over seeds on the test split, best-validation checkpoint. RMSE per seed = sqrt(test MSE).

The per-epoch curves use validation MSE: the runs log train loss and validation MSE each epoch but evaluate the test split only once, at the best checkpoint.

## Test MSE

| task | QLSTM | Q-sLSTM | Q-sLSTM (fk) | best |
|---|---|---|---|---|
| delay | 0.2491 ± 0.0054 | 0.2532 ± 0.0123 | 0.0020 ± 0.0007 | Q-sLSTM (fk) |
| ema | 0.0007 ± 0.0003 | 0.0040 ± 0.0012 | 0.0000 ± 0.0000 | Q-sLSTM (fk) |
| running_max | 0.0180 ± 0.0024 | 0.0102 ± 0.0018 | 0.0004 ± 0.0003 | Q-sLSTM (fk) |
| flip_flop | 0.1308 ± 0.0127 | 0.1526 ± 0.0468 | 0.0000 ± 0.0000 | Q-sLSTM (fk) |
| narma | 0.0069 ± 0.0004 | 0.0068 ± 0.0004 | 0.0045 ± 0.0007 | Q-sLSTM (fk) |
| sine_next | 0.0137 ± 0.0010 | 0.0102 ± 0.0010 | 0.0001 ± 0.0001 | Q-sLSTM (fk) |

## Test MAE

| task | QLSTM | Q-sLSTM | Q-sLSTM (fk) | best |
|---|---|---|---|---|
| delay | 0.4145 ± 0.0064 | 0.4225 ± 0.0127 | 0.0342 ± 0.0059 | Q-sLSTM (fk) |
| ema | 0.0190 ± 0.0051 | 0.0441 ± 0.0080 | 0.0034 ± 0.0007 | Q-sLSTM (fk) |
| running_max | 0.0956 ± 0.0058 | 0.0768 ± 0.0078 | 0.0143 ± 0.0026 | Q-sLSTM (fk) |
| flip_flop | 0.2438 ± 0.0146 | 0.2808 ± 0.0567 | 0.0037 ± 0.0014 | Q-sLSTM (fk) |
| narma | 0.0651 ± 0.0020 | 0.0645 ± 0.0020 | 0.0518 ± 0.0044 | Q-sLSTM (fk) |
| sine_next | 0.0904 ± 0.0033 | 0.0755 ± 0.0039 | 0.0076 ± 0.0017 | Q-sLSTM (fk) |

## Test RMSE

| task | QLSTM | Q-sLSTM | Q-sLSTM (fk) | best |
|---|---|---|---|---|
| delay | 0.4990 ± 0.0054 | 0.5031 ± 0.0123 | 0.0439 ± 0.0073 | Q-sLSTM (fk) |
| ema | 0.0254 ± 0.0060 | 0.0619 ± 0.0115 | 0.0055 ± 0.0011 | Q-sLSTM (fk) |
| running_max | 0.1338 ± 0.0085 | 0.1003 ± 0.0095 | 0.0203 ± 0.0061 | Q-sLSTM (fk) |
| flip_flop | 0.3613 ± 0.0171 | 0.3839 ± 0.0738 | 0.0065 ± 0.0023 | Q-sLSTM (fk) |
| narma | 0.0832 ± 0.0026 | 0.0823 ± 0.0024 | 0.0670 ± 0.0052 | Q-sLSTM (fk) |
| sine_next | 0.1169 ± 0.0043 | 0.1011 ± 0.0047 | 0.0105 ± 0.0023 | Q-sLSTM (fk) |

## Best validation MSE and epoch

| task | QLSTM | Q-sLSTM | Q-sLSTM (fk) |
|---|---|---|---|
| delay | 0.2513 ± 0.0075 (ep 60 ± 1) | 0.2552 ± 0.0117 (ep 59 ± 1) | 0.0020 ± 0.0006 (ep 58 ± 2) |
| ema | 0.0007 ± 0.0003 (ep 60 ± 1) | 0.0039 ± 0.0012 (ep 59 ± 1) | 0.0000 ± 0.0000 (ep 56 ± 3) |
| running_max | 0.0175 ± 0.0017 (ep 59 ± 1) | 0.0098 ± 0.0016 (ep 59 ± 1) | 0.0004 ± 0.0003 (ep 59 ± 1) |
| flip_flop | 0.1300 ± 0.0120 (ep 57 ± 4) | 0.1535 ± 0.0452 (ep 59 ± 2) | 0.0000 ± 0.0000 (ep 60 ± 1) |
| narma | 0.0070 ± 0.0007 (ep 59 ± 2) | 0.0069 ± 0.0007 (ep 59 ± 1) | 0.0046 ± 0.0009 (ep 56 ± 4) |
| sine_next | 0.0135 ± 0.0013 (ep 59 ± 1) | 0.0101 ± 0.0011 (ep 59 ± 1) | 0.0001 ± 0.0001 (ep 59 ± 1) |

## Paired test-MSE differences (a − b; negative = a lower)

Wilcoxon signed-rank over seeds; `wins` = seeds where a has lower MSE.

| task | a − b | mean diff ± sd | wins | p |
|---|---|---|---|---|
| delay | Q-sLSTM − QLSTM | +0.0041 ± 0.0118 | 8/20 | 0.143 |
| delay | Q-sLSTM (fk) − QLSTM | -0.2471 ± 0.0053 | 20/20 | 1.91e-06 |
| delay | Q-sLSTM − Q-sLSTM (fk) | +0.2512 ± 0.0121 | 0/20 | 1.91e-06 |
| ema | Q-sLSTM − QLSTM | +0.0033 ± 0.0012 | 0/20 | 1.91e-06 |
| ema | Q-sLSTM (fk) − QLSTM | -0.0007 ± 0.0003 | 20/20 | 1.91e-06 |
| ema | Q-sLSTM − Q-sLSTM (fk) | +0.0039 ± 0.0012 | 0/20 | 1.91e-06 |
| running_max | Q-sLSTM − QLSTM | -0.0078 ± 0.0034 | 20/20 | 1.91e-06 |
| running_max | Q-sLSTM (fk) − QLSTM | -0.0175 ± 0.0024 | 20/20 | 1.91e-06 |
| running_max | Q-sLSTM − Q-sLSTM (fk) | +0.0097 ± 0.0019 | 0/20 | 1.91e-06 |
| flip_flop | Q-sLSTM − QLSTM | +0.0217 ± 0.0463 | 4/20 | 0.04 |
| flip_flop | Q-sLSTM (fk) − QLSTM | -0.1308 ± 0.0127 | 20/20 | 1.91e-06 |
| flip_flop | Q-sLSTM − Q-sLSTM (fk) | +0.1525 ± 0.0468 | 0/20 | 1.91e-06 |
| narma | Q-sLSTM − QLSTM | -0.0001 ± 0.0007 | 12/20 | 0.216 |
| narma | Q-sLSTM (fk) − QLSTM | -0.0024 ± 0.0008 | 20/20 | 1.91e-06 |
| narma | Q-sLSTM − Q-sLSTM (fk) | +0.0023 ± 0.0007 | 0/20 | 1.91e-06 |
| sine_next | Q-sLSTM − QLSTM | -0.0035 ± 0.0013 | 20/20 | 1.91e-06 |
| sine_next | Q-sLSTM (fk) − QLSTM | -0.0136 ± 0.0010 | 20/20 | 1.91e-06 |
| sine_next | Q-sLSTM − Q-sLSTM (fk) | +0.0101 ± 0.0010 | 0/20 | 1.91e-06 |

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
