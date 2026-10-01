# QLSTM vs Q-sLSTM — sequence length 32 vs 16

Sweeps: `2026-09-28` (sequence_length=32) and `2026-09-30` (sequence_length=16; commit 65850e4b is labelled "result length 8" but every config.json says 16). Same 20 seeds and preset otherwise, best-validation checkpoint, 45 trainable parameters each.

Values are mean ± SD over seeds. *Q-sLSTM lower* = seeds where Q-sLSTM has the lower value; p = Wilcoxon signed-rank, paired by seed, unadjusted.

**Excluded:** narma, L=16, seed 1106723791. One of its 400 test sequences (id 1000274) has a generated target that diverges to ~29 (typical |y| ≈ 0.1). Both models score MSE ≈ 181 on that sequence, which alone lifts the seed's test MSE to 0.46 and the task SD to 0.10. It is a NARMA generator defect, not model behaviour, so narma L=16 uses n = 19.

## Test MSE (`fig_mse.png`)

| task | L | QLSTM | Q-sLSTM | Q-sLSTM lower | p |
|---|---|---|---|---|---|
| delay | 32 | 0.249 ± 0.005 | 0.250 ± 0.011 | 10/20 | 0.84 |
| | 16 | 0.248 ± 0.007 | 0.254 ± 0.016 | 8/20 | 0.14 |
| ema | 32 | 0.00068 ± 0.00033 | 0.0040 ± 0.0016 | 0/20 | 2e-6 |
| | 16 | 0.00064 ± 0.00038 | 0.0044 ± 0.0030 | 0/20 | 2e-6 |
| running_max | 32 | 0.0180 ± 0.0024 | 0.0030 ± 0.0008 | 20/20 | 2e-6 |
| | 16 | 0.0221 ± 0.0034 | 0.0028 ± 0.0007 | 20/20 | 2e-6 |
| flip_flop | 32 | 0.131 ± 0.013 | 0.105 ± 0.014 | 19/20 | 4e-6 |
| | 16 | 0.069 ± 0.007 | 0.056 ± 0.008 | 19/20 | 1e-5 |
| narma | 32 | 0.0069 ± 0.0004 | 0.0057 ± 0.0004 | 20/20 | 2e-6 |
| | 16 | 0.0072 ± 0.0005 | 0.0055 ± 0.0006 | 19/19 | 4e-6 |
| sine_next | 32 | 0.0137 ± 0.0010 | 0.0075 ± 0.0015 | 20/20 | 2e-6 |
| | 16 | 0.0145 ± 0.0019 | 0.0077 ± 0.0013 | 20/20 | 2e-6 |

RMSE (`fig_rmse.png`) and MAE (`fig_mae.png`) give the same winner on every task and length. The one exception is flip_flop MAE at L=16, where the models tie (10/20, p = 0.41) even though Q-sLSTM has lower MSE/RMSE on 19/20 seeds. At that length, Q-sLSTM's advantage on flip_flop comes from fewer large errors rather than lower typical error.

**Halving the length leaves the picture unchanged.** Q-sLSTM wins running_max, flip_flop, narma and sine_next and loses ema at both lengths. Neither model learns delay (MSE ≈ target variance × 0.75). QLSTM gets worse on running_max at L=16 (0.018 → 0.022) while Q-sLSTM does not, so that gap widens. Both models roughly halve their flip_flop error at L=16.

## Generalization gap (`fig_generalization_gap.png`)

Gap = test MSE − train MSE, both from the best-validation checkpoint. The runs do not log train MSE, so each checkpoint was replayed on its full 1400-sequence training split (inference only; the replayed test MSE matches the logged one to < 0.1%; cached in `replay_train_mse.csv`).

| task | L | QLSTM gap | Q-sLSTM gap | p |
|---|---|---|---|---|
| delay | 32 | −0.0012 ± 0.0031 | −0.0014 ± 0.0029 | 0.50 |
| | 16 | −0.0004 ± 0.0052 | +0.0001 ± 0.0052 | 0.33 |
| ema | 32 | −0.00001 ± 0.00003 | +0.00000 ± 0.00015 | 0.67 |
| | 16 | −0.00001 ± 0.00005 | −0.00005 ± 0.00028 | 0.52 |
| running_max | 32 | +0.0001 ± 0.0010 | +0.00003 ± 0.00015 | 0.70 |
| | 16 | −0.0001 ± 0.0010 | −0.00000 ± 0.00017 | 0.57 |
| flip_flop | 32 | −0.0021 ± 0.0096 | −0.0012 ± 0.0070 | 0.67 |
| | 16 | −0.0024 ± 0.0042 | −0.0014 ± 0.0043 | 0.47 |
| narma | 32 | −0.0004 ± 0.0018 | −0.0004 ± 0.0018 | 0.19 |
| | 16 | −0.0002 ± 0.0007 | −0.00004 ± 0.0007 | 0.045 |
| sine_next | 32 | −0.0001 ± 0.0004 | −0.00003 ± 0.0003 | 0.19 |
| | 16 | −0.0001 ± 0.0004 | −0.00006 ± 0.0002 | 0.15 |

**No overfitting at either length.** Every mean gap is within one SD of zero, and most are slightly negative (test ≤ train). Q-sLSTM's gap is never significantly different from QLSTM's after accounting for six tasks (narma L=16, p = 0.045, does not survive correction). With 45 parameters and 1400 training sequences, both models are capacity-limited, not data-limited, so their test-error differences reflect fit, not generalization. This matches the best epoch being ~60 for almost every run.

## Before the length-8 run

NARMA's warm-up is 10 steps (`narma_order`), so at L=16 only 6 steps per sequence are scored, and at L=8 there are none: `ScalarTaskConfig` raises an error when `sequence_length <= warm-up` (`src/q_slstm/datasets/scalar_tasks.py:65`). Delay (warm-up 4) would be scored on only 4 steps.

## Files

- `fig_mse.png`, `fig_rmse.png`, `fig_mae.png`, `fig_generalization_gap.png`
- `summary_by_length.csv` — mean, SD, wins, p per length × task × metric (mse, rmse, mae, train_mse, gen_gap)
- `metrics_per_seed.csv` — every seed, both lengths, both models; `excluded` marks the dropped seed
- `replay_train_mse.csv` — cached replay results
- `len_compare.py` — regenerates everything (run from the repo root); add a length by adding it to `SWEEPS`
