"""Masked loss and metric functions: exclusions, hand-calculated values, paired comparison."""

import numpy as np
import pandas as pd
import pytest
import torch

from q_slstm.experiments import nearest_neighbor_metrics as nnm
from q_slstm.experiments.nearest_neighbor import masked_mse

# Tokens: 0 reference, 1 first candidate, 2..5 candidates; a record at token 4 (global best).
PRED = np.array([[9.0, 9.0, 0.5, 0.2, 0.8, 0.8]])
TARGET = np.array([[0.0, 0.1, 0.5, 0.5, 0.9, 0.9]])
EVENT = np.array([[0, 1, 0, 0, 1, 0]], bool)
METRIC = np.array([[0, 0, 1, 1, 1, 1]], bool)
ALPHA = np.array([[0.9, 0.9, 0.1, 0.2, 0.6, 0.3]])


def test_training_loss_ignores_reference_but_includes_first_candidate():
    targets = torch.zeros(1, 4, 1)
    loss_mask = torch.tensor([[False, True, True, True]]).unsqueeze(-1)
    base = torch.tensor([[0.0, 1.0, 0.0, 0.0]]).unsqueeze(-1)
    bad_reference = base.clone()
    bad_reference[0, 0, 0] = 1e6
    bad_first = base.clone()
    bad_first[0, 1, 0] = 3.0
    assert masked_mse(bad_reference, targets, loss_mask) == masked_mse(base, targets, loss_mask)
    assert masked_mse(base, targets, loss_mask).item() == pytest.approx(1 / 3)
    assert masked_mse(bad_first, targets, loss_mask).item() == pytest.approx(9 / 3)  # first candidate counts


def test_masked_mse_checks_shapes():
    with pytest.raises(ValueError, match="shape mismatch"):
        masked_mse(torch.zeros(1, 3, 1), torch.zeros(1, 3), torch.ones(1, 3, 1, dtype=torch.bool))


def test_metrics_match_hand_calculation():
    row = nnm.sequence_metrics(PRED, TARGET, EVENT, METRIC, ALPHA).iloc[0]
    # metric steps t=2..5: errors 0, .3, .1, .1
    assert row["n_metric_steps"] == 4 and row["n_events"] == 1 and row["n_nonevents"] == 3
    assert row["mse"] == pytest.approx((0 + 0.09 + 0.01 + 0.01) / 4)
    assert row["mae"] == pytest.approx(0.125)
    assert row["final_mse"] == pytest.approx(0.01) and row["final_mae"] == pytest.approx(0.1)
    assert row["event_mse"] == pytest.approx(0.01) and row["nonevent_mse"] == pytest.approx(0.1 / 3)
    # drift needs both t and t-1 to be metric steps: non-events t=3 (|.2-.5|) and t=5 (0); t=2 skipped
    assert row["n_drift_steps"] == 2 and row["drift_nonevent"] == pytest.approx(0.15)
    assert row["drift_post_best"] == pytest.approx(0.0)  # only t=5 follows the global-best event
    # revision gain at t=4: |pred_3 - target_4| - |pred_4 - target_4| = 0.7 - 0.1
    assert row["revision_gain"] == pytest.approx(0.6) and row["n_revision_steps"] == 1
    # event-aligned error: lag 0 -> t=4, lag 1 -> t=5, lag -1 -> t=3; later lags unavailable => NaN
    assert row["event_abs_err_lag_0"] == pytest.approx(0.1) and row["event_abs_err_lag_1"] == pytest.approx(0.1)
    assert row["event_abs_err_lag_m1"] == pytest.approx(0.3)
    assert np.isnan(row["event_abs_err_lag_2"]) and np.isnan(row["best_abs_err_lag_16"])
    # alpha
    assert row["alpha_event"] == pytest.approx(0.6) and row["alpha_nonevent"] == pytest.approx(0.2)
    assert row["alpha_contrast"] == pytest.approx(0.4)
    assert row["event_alpha_lag_1"] == pytest.approx(0.3) and np.isnan(row["event_alpha_lag_2"])


def test_metrics_are_unchanged_by_reference_and_first_candidate_predictions():
    altered_pred, altered_alpha = PRED.copy(), ALPHA.copy()
    altered_pred[0, :2] = [-1e6, 1e6]
    altered_alpha[0, :2] = [123.0, -5.0]
    a = nnm.sequence_metrics(PRED, TARGET, EVENT, METRIC, ALPHA)
    b = nnm.sequence_metrics(altered_pred, TARGET, EVENT, METRIC, altered_alpha)
    pd.testing.assert_frame_equal(a, b)


def test_missing_groups_are_nan_not_zero():
    pred = np.array([[0, 0, 0.5, 0.5, 0.5]], float)
    target = np.array([[0, 0.5, 0.5, 0.5, 0.5]], float)
    event = np.array([[0, 1, 0, 0, 0]], bool)  # only the (excluded) first-candidate event
    metric = np.array([[0, 0, 1, 1, 1]], bool)
    row = nnm.sequence_metrics(pred, target, event, metric, np.full((1, 5), 0.4)).iloc[0]
    assert row["n_events"] == 0
    for col in ("event_mse", "event_mae", "revision_gain", "alpha_event", "alpha_contrast", "event_abs_err_lag_0"):
        assert np.isnan(row[col]), col
    assert row["nonevent_mse"] == 0.0 and row["alpha_nonevent"] == pytest.approx(0.4)
    assert row["drift_post_best"] == 0.0  # global best is the first candidate; drift at t=3,4 only
    assert row["n_drift_steps"] == 2


def test_aggregation_counts_and_excludes_undefined_sequences():
    seq = nnm.sequence_metrics(np.vstack([PRED, PRED]), np.vstack([TARGET, TARGET]),
                               np.vstack([EVENT, np.zeros_like(EVENT)]), np.vstack([METRIC, METRIC]),
                               np.vstack([ALPHA, ALPHA]))
    seq = nnm.add_identity(seq, 0, "qlstm", "test", ["iid", "late"], [10, 11])
    agg = nnm.aggregate_per_seed(seq)
    row = agg[agg["case_type"] == "all"].iloc[0]
    assert row["n_sequences"] == 2 and row["n_events"] == 1
    assert row["event_mse__n"] == 1  # the sequence without events does not contribute a zero
    assert row["event_mse"] == pytest.approx(0.01)
    assert set(agg["case_type"]) == {"all", "iid", "late"}


def test_paired_difference_uses_seed_level_differences():
    rows = []
    for seed, (a, b) in enumerate([(0.10, 0.15), (0.20, 0.22), (0.05, 0.11)]):
        for model, v in (("qslstm", a), ("qlstm", b)):
            rows.append({"run_seed": seed, "model": model, "split": "test", "case_type": "all", "mse": v})
    table, summary = nnm.paired_difference(pd.DataFrame(rows))
    np.testing.assert_allclose(table["difference"], [-0.05, -0.02, -0.06])
    assert summary["n_paired_seeds"] == 3 and summary["mean_difference"] == pytest.approx(-0.13 / 3)
    half = nnm.t_critical_95(2) * np.std([-0.05, -0.02, -0.06], ddof=1) / np.sqrt(3)
    assert summary["ci95_high"] - summary["ci95_low"] == pytest.approx(2 * half)
    assert summary["seeds_a_lower"] == 3
