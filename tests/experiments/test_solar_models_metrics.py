"""Projected quantum models, paired initialization, final-step loss, metrics, and alpha diagnostics."""

import math

import numpy as np
import pandas as pd
import pytest
import torch

from q_slstm.models.projected_quantum import build_projected_quantum_model
from q_slstm.models.sequence_wrappers import CustomLSTM, CustomQsLSTM
from qslstm_test_utils import make_cell, make_qlstm_cell
from q_slstm.experiments import solar_generation_metrics as sm
from q_slstm.experiments.solar_generation import derive_seeds, final_step_mse

B, L = 3, 5


def _pair(seed=0, hidden=2, depth=1):
    seeds = derive_seeds(seed)
    return [build_projected_quantum_model(m, 13, 3, hidden, 1, depth, model_seed=seeds["model"],
                                          projection_seed=seeds["projection"]) for m in ("qlstm", "qslstm")]


# ---------------------------------------------------------------------------------------------
# 8. Paired initialization
# ---------------------------------------------------------------------------------------------

def test_paired_models_have_equal_but_independent_initial_parameters_including_projection():
    rng_state = torch.get_rng_state()
    qlstm, qslstm = _pair()
    assert torch.equal(rng_state, torch.get_rng_state())  # isolated RNG contexts
    assert isinstance(qlstm.core, CustomLSTM) and isinstance(qslstm.core, CustomQsLSTM)
    sd_a, sd_b = qlstm.state_dict(), qslstm.state_dict()
    assert sd_a.keys() == sd_b.keys() and "projection.weight" in sd_a
    for name in sd_a:
        assert torch.equal(sd_a[name], sd_b[name]), name
        assert sd_a[name].data_ptr() != sd_b[name].data_ptr(), name
    assert qlstm.n_qubits == qslstm.n_qubits == 3 + 2
    assert qlstm.projection.weight.shape == (3, 13)
    with torch.no_grad():
        qlstm.projection.weight.add_(1.0)
    assert not torch.equal(qlstm.projection.weight, qslstm.projection.weight)
    other = _pair(seed=1)[0]
    assert not torch.equal(other.projection.weight, qslstm.projection.weight)
    seeds = derive_seeds(0)
    assert len({seeds["model"], seeds["projection"], seeds["loader"]}) == 3 and seeds["data_seed"] is None


# ---------------------------------------------------------------------------------------------
# 9. Real projected forward / backward and final-step loss
# ---------------------------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("index", [0, 1])
def test_real_projected_quantum_forward_backward(index):
    model = _pair()[index]
    x = torch.randn(B, L, 13, generator=torch.Generator().manual_seed(0))
    outputs = model(x)[0]
    assert outputs.shape == (B, L, 1)
    target = torch.randn(B, 1)
    loss = final_step_mse(outputs, target)
    loss.backward()
    assert torch.isfinite(loss)
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    assert model.projection.weight.grad.abs().sum() > 0
    assert model.core.cell.input_gate.weights.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="13"):
        model(torch.randn(B, L, 3))


def test_final_step_loss_ignores_earlier_outputs_and_rejects_broadcasting():
    outputs = torch.zeros(2, 4, 1, requires_grad=True)
    target = torch.tensor([[1.0], [3.0]])
    loss = final_step_mse(outputs, target)
    assert loss.item() == pytest.approx(5.0)
    loss.backward()
    assert outputs.grad[:, :-1].abs().sum() == 0 and outputs.grad[:, -1].abs().sum() > 0
    with pytest.raises(ValueError, match=r"\[B, 1\]"):
        final_step_mse(outputs, torch.tensor([1.0, 3.0]))  # [B] would broadcast to [B, B]


# ---------------------------------------------------------------------------------------------
# 10. Metrics
# ---------------------------------------------------------------------------------------------

def test_point_metrics_match_hand_calculation():
    m = sm.point_metrics([0.0, 10.0, 20.0], [1.0, 8.0, 23.0])  # errors 1, -2, 3
    assert m["n"] == 3 and m["mse"] == pytest.approx(14 / 3) and m["mae"] == pytest.approx(2.0)
    assert m["rmse"] == pytest.approx(math.sqrt(14 / 3))
    assert m["r2"] == pytest.approx(1 - 14 / 200)
    assert sm.point_metrics([5.0, 5.0], [4.0, 6.0])["r2"] is None  # constant target
    assert sm.point_metrics([], [])["n"] == 0


def _pred_frame(model="qlstm", seed=0, offset=0.0):
    return pd.DataFrame({
        "run_seed": seed, "model": model, "dataset_id": "d", "window_id": [1, 2, 3, 4],
        "target_mwh": [0.0, 10.0, 20.0, 0.0], "prediction_mwh": np.array([-1.0, 12.0, 17.0, 0.0]) + offset,
        "persistence_mwh": [0.0, 0.0, 10.0, 20.0], "daily_persistence_mwh": [np.nan, 9.0, 21.0, 0.0],
        "target_month": 6, "target_hour_est": [3, 9, 10, 20], "target_date_est": ["2024-06-01"] * 2 + ["2024-06-02"] * 2,
        "is_daylight_proxy": pd.array([False, True, True, pd.NA], dtype="boolean"),
        "is_large_ramp": [False, True, True, True],
    })


def test_seed_metrics_groups_baseline_subset_and_missing_daylight():
    table = sm.seed_metrics(_pred_frame())
    get = lambda **k: table[np.logical_and.reduce([table[c] == v for c, v in k.items()])].iloc[0]
    overall = get(subset="all_eligible", group_type="overall", forecaster="model")
    assert overall["n"] == 4 and overall["mse"] == pytest.approx((1 + 4 + 9 + 0) / 4)
    assert overall["n_negative_predictions"] == 1  # negative prediction kept, not clamped
    pers = get(subset="all_eligible", group_type="overall", forecaster="persistence")
    assert pers["mse"] == pytest.approx((0 + 100 + 100 + 400) / 4)
    common = get(subset="baseline_common", group_type="overall", forecaster="daily_persistence")
    assert common["n"] == 3 and common["mse"] == pytest.approx((1 + 1 + 0) / 3)  # NaN baseline excluded
    common_model = get(subset="baseline_common", group_type="overall", forecaster="model")
    assert common_model["n"] == 3 and common_model["mse"] == pytest.approx((4 + 9 + 0) / 3)
    groups = set(table[(table["group_type"] == "daylight_proxy") & (table["subset"] == "all_eligible")]["group"])
    assert groups == {"True", "False", "missing"}
    assert get(subset="all_eligible", group_type="daylight_proxy", group="missing", forecaster="model")["n"] == 1
    empty_night = sm.seed_metrics(_pred_frame().assign(is_daylight_proxy=pd.array([True] * 4, dtype="boolean")))
    assert set(empty_night[empty_night["group_type"] == "daylight_proxy"]["group"]) == {"True"}  # no empty group rows
    daily = sm.daily_metrics(_pred_frame())
    assert daily["model_mse"].tolist() == pytest.approx([2.5, 4.5])


def test_paired_differences_over_seeds():
    frames = [sm.seed_metrics(_pred_frame(m, s, off)) for m, s, off in
              [("qlstm", 0, 0.0), ("qslstm", 0, 1.0), ("qlstm", 1, 0.0), ("qslstm", 1, -1.0), ("qlstm", 2, 0.0)]]
    table, summary = sm.paired_difference(pd.concat(frames))
    assert table["run_seed"].tolist() == [0, 1]  # seed 2 is unpaired
    # offsets shift errors by +/-1: MSE(errors -1,2,-3,0 shifted) computed directly
    base = np.array([-1.0, 2.0, -3.0, 0.0])
    expected = [np.mean((base + 1) ** 2) - np.mean(base**2), np.mean((base - 1) ** 2) - np.mean(base**2)]
    assert table["difference"].tolist() == pytest.approx(expected)
    assert summary["n_paired_seeds"] == 2 and summary["mean_difference"] == pytest.approx(np.mean(expected))
    assert summary["sd_difference"] == pytest.approx(np.std(expected, ddof=1))
    one = sm.paired_difference(pd.concat(frames[:2]))[1]
    assert one["n_paired_seeds"] == 1 and math.isnan(one["sd_difference"])


# ---------------------------------------------------------------------------------------------
# 11. Alpha diagnostics
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["qlstm", "qslstm"])
def test_diagnostics_preserve_predictions_and_gradients(kind):
    torch.manual_seed(0)
    core = CustomLSTM(3, 2, make_qlstm_cell(3, 2)) if kind == "qlstm" else CustomQsLSTM(3, 2, make_cell(3, 2))
    from q_slstm.models.projected_quantum import ProjectedQuantumModel

    model = ProjectedQuantumModel(torch.nn.Linear(13, 3), core, 13, 3)
    x = torch.randn(B, L, 13)
    plain = model(x)[0]
    plain[:, -1].sum().backward()
    grads = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    diagnosed, _, diag = model(x, return_diagnostics=True)
    diagnosed[:, -1].sum().backward()
    assert torch.equal(plain, diagnosed)
    assert all(torch.equal(a, p.grad) for a, p in zip(grads, model.parameters()))
    alpha = diag["alpha"]
    assert alpha.shape == (B, L, 2) and not alpha.requires_grad
    assert ((alpha >= 0) & (alpha <= 1 + 1e-6)).all()


def test_alpha_curves_reduce_after_elementwise_ratio_and_exclude_first_step():
    alpha_steps = np.array([[1.0, 0.2, 0.4], [1.0, 0.6, 0.8]])
    curves = sm.alpha_curves(alpha_steps, pd.array([True, False], dtype="boolean"), {"run_seed": 0, "model": "x"})
    assert set(curves["input_step"]) == {1, 2}  # step 0 (zero-initialized normalizer) is excluded
    get = lambda g, s: curves[(curves["target_group"] == g) & (curves["input_step"] == s)]["alpha_mean"].item()
    assert get("all", 1) == pytest.approx(0.4) and get("daylight", 2) == pytest.approx(0.4)
    assert get("night", 2) == pytest.approx(0.8)
    assert not any("gate" in c for c in curves.columns)
