"""Independent gate/recurrence references and integration checks for qslstm_log."""

import math

import pytest
import torch

from q_slstm.models.factory import build_quantum_model
from q_slstm.models.q_slstm_log_cell import (
    QSLSTM_LOG_RECURRENCE,
    CustomQsLSTMLogCell,
    float64_expectations,
    logarithmic_gate,
    logarithmic_memory_update,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_gate_values_and_derivatives_including_boundaries(dtype):
    near_one = torch.nextafter(torch.ones((), dtype=dtype), torch.zeros((), dtype=dtype)).item()
    q = torch.tensor([-1, -near_one, -.5, 0, .5, near_one], dtype=dtype, requires_grad=True)
    actual = logarithmic_gate(q)
    # A float64 scalar reference avoids losing the tiny gate beside -1 in float32.
    expected = torch.tensor([math.log1p((1 + x) / (1 - x)) for x in q.tolist()], dtype=dtype)
    torch.testing.assert_close(actual, expected, atol=0, rtol=2 * torch.finfo(dtype).eps)
    actual.sum().backward()
    torch.testing.assert_close(q.grad, 1 / (1 - q.detach()))
    assert actual[0] == 0
    assert actual[1] > 0
    assert actual[3].item() == pytest.approx(math.log(2))


@pytest.mark.parametrize("bad,match", [
    (1., "singular"), (1.01, "outside"), (-1.01, "outside"),
    (float("nan"), "NaN"), (float("inf"), "infinity"),
])
def test_invalid_expectations_raise(bad, match):
    with pytest.raises(ValueError, match=match):
        logarithmic_gate(torch.tensor([bad]))


def test_multistep_values_states_diagnostics_and_gradients_match_direct_recurrence():
    gen = torch.Generator().manual_seed(214)
    qi = (torch.rand(24, 3, generator=gen, dtype=torch.float64) * 1.8 - .9).requires_grad_()
    qf = (torch.rand(24, 3, generator=gen, dtype=torch.float64) * 1.8 - .9).requires_grad_()
    z = torch.randn(24, 3, generator=gen, dtype=torch.float64).tanh().requires_grad_()
    c = n = scale = torch.zeros(3, dtype=torch.float64)
    direct_c = direct_n = torch.zeros_like(c)
    actual, expected = [], []
    for t in range(len(qi)):
        i, f = torch.log(2 / (1 - qi[t])), torch.log(2 / (1 - qf[t]))
        direct_c, direct_n = f * direct_c + i * z[t], f * direct_n + i
        old_c, old_n = c, n
        c, n, scale, iw, fw = logarithmic_memory_update(
            qi[t], qf[t], z[t], c, n, scale, return_forget_weight=True)
        torch.testing.assert_close(c, fw * old_c + iw * z[t])
        torch.testing.assert_close(n, fw * old_n + iw)
        torch.testing.assert_close(torch.ldexp(c, scale.long()), direct_c)
        torch.testing.assert_close(torch.ldexp(n, scale.long()), direct_n)
        torch.testing.assert_close(iw / n, i / direct_n)
        actual.append(c / n)
        expected.append(direct_c / direct_n)
    actual, expected = torch.stack(actual), torch.stack(expected)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    got = torch.autograd.grad(actual.square().sum(), (qi, qf, z))
    want = torch.autograd.grad(expected.square().sum(), (qi, qf, z))
    for a, b in zip(got, want):
        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("qi,qf", [(-1., .2), (.2, -1.)])
def test_zero_gate_boundary_preserves_correct_values_and_gradients(qi, qf):
    qi, qf = (torch.tensor([v], dtype=torch.float64, requires_grad=True) for v in (qi, qf))
    c, n, z, scale = (torch.tensor([v], dtype=torch.float64) for v in (.3, .8, -.7, 3.))
    ct, nt, _, _ = logarithmic_memory_update(qi, qf, z, c, n, scale)
    i, f = torch.log(2 / (1 - qi)), torch.log(2 / (1 - qf))
    direct = (f * c * 8 + i * z) / (f * n * 8 + i)
    torch.testing.assert_close(ct / nt, direct)
    got = torch.autograd.grad((ct / nt).sum(), (qi, qf))
    want = torch.autograd.grad(direct.sum(), (qi, qf))
    for a, b in zip(got, want):
        torch.testing.assert_close(a, b)


def test_empty_memory_with_zero_write_raises():
    zero = torch.zeros(1)
    with pytest.raises(ValueError, match="zero denominator"):
        logarithmic_memory_update(zero - 1, zero, zero, zero, zero, zero)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_long_amplification_keeps_stored_states_bounded(dtype):
    # With the amplified forget gate N grows without bound; binary scaling carries it.
    qi, qf, z = (torch.tensor([v], dtype=dtype) for v in (.2, .99, .25))
    c = n = scale = torch.zeros(1, dtype=dtype)
    for _ in range(600):
        c, n, scale, _ = logarithmic_memory_update(qi, qf, z, c, n, scale, amplified_forget=True)
        assert .5 <= n.item() < 1
        torch.testing.assert_close(c / n, z)
    assert scale.item() > 1024


def test_factory_preserves_parameter_initialization_and_selects_new_cell():
    before = torch.get_rng_state()
    original = build_quantum_model("qslstm", 1, 2, 1, 1, seed=71)
    variant = build_quantum_model("qslstm_log", 1, 2, 1, 1, seed=71)
    assert torch.equal(before, torch.get_rng_state())
    assert isinstance(variant.cell, CustomQsLSTMLogCell)
    assert variant.cell.recurrence == QSLSTM_LOG_RECURRENCE
    assert original.state_dict().keys() == variant.state_dict().keys()
    for key, value in original.state_dict().items():
        assert torch.equal(value, variant.state_dict()[key])


def test_nearest_neighbor_config_records_log_recurrence():
    from q_slstm.experiments.nearest_neighbor import resolve_config

    config = resolve_config({"model": "qslstm_log", "scale": "pilot", "seed": 0})
    assert config["qslstm_recurrence"] == QSLSTM_LOG_RECURRENCE


def test_real_vqc_rollout_chunking_and_backward():
    model = build_quantum_model("qslstm_log", 1, 2, 1, 1, seed=21)
    x = torch.tensor([[[.1], [.4], [-.2]], [[.3], [-.5], [.6]]])
    outputs, state, diag = model(x, return_diagnostics=True)
    first, initial = model(x[:, :1])
    rest, final = model(x[:, 1:], initial)
    torch.testing.assert_close(outputs, torch.cat((first, rest), dim=1))
    for a, b in zip(state, final):
        torch.testing.assert_close(a, b)
    assert diag["alpha"].shape == (2, 3, 2)
    assert ((diag["alpha"] >= 0) & (diag["alpha"] <= 1)).all()
    outputs.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_legacy_sigmoid_forget_normalizer_converges_to_input_over_one_minus_forget(dtype):
    qi, qf, z = (torch.tensor([v], dtype=dtype) for v in (.2, .99, .25))
    c = n = scale = torch.zeros(1, dtype=dtype)
    for _ in range(200):
        c, n, scale, _ = logarithmic_memory_update(qi, qf, z, c, n, scale, amplified_forget=False)
        torch.testing.assert_close(c / n, z)
    i, f = torch.log(2 / (1 - qi)), torch.sigmoid(qf)
    torch.testing.assert_close(torch.ldexp(n, scale.long()), i / (1 - f))


def test_input_and_forget_gates_are_amplified_by_default():
    qi, qf = torch.tensor([.9]), torch.tensor([.7])
    c, n, scale = torch.tensor([.5]), torch.tensor([1.]), torch.zeros(1)
    z = torch.tensor([0.])
    ct, nt, st, iw, fw = logarithmic_memory_update(qi, qf, z, c, n, scale, return_forget_weight=True)
    torch.testing.assert_close(torch.ldexp(fw, st.long()), torch.log(2 / (1 - qf)))
    torch.testing.assert_close(torch.ldexp(iw, st.long()), torch.log(2 / (1 - qi)))
    ct, nt, st, iw, fw = logarithmic_memory_update(qi, qf, z, c, n, scale, return_forget_weight=True,
                                                   amplified_forget=False)
    torch.testing.assert_close(torch.ldexp(fw, st.long()), torch.sigmoid(qf))


def _boundary_input_gate(cell):
    """Input-gate weights whose t=0 expectations lie within float32 rounding of +-1.

    With h_prev = 0 the hidden qubits stay in |+>, so their q depends only on the last rotation;
    offsets of 1e-5 / 3e-4 from +-pi/2 make the float32 simulation return exactly +-1 or overshoot
    to +-1.0000001. Unit 0 is entangled with the input and stays away from the boundary.
    """
    with torch.no_grad():
        w = cell.input_gate.weights
        w.zero_()
        w[-1, 1:4] = torch.tensor([math.pi / 2 + 1e-5, -math.pi / 2 - 1e-5, math.pi / 2 + 3e-4])


def test_float32_expectations_hit_the_boundary_that_crashed_training():
    cell = build_quantum_model("qslstm_log", 1, 4, 1, 2, seed=0).cell
    _boundary_input_gate(cell)
    combined = torch.cat((torch.full((8, 1), .3), torch.zeros(8, 4)), dim=-1)
    q32 = cell.input_gate(combined)
    assert (q32.abs() >= 1).any(), "float32 path should land on or past +-1 (the old failure mode)"
    with pytest.raises(ValueError):
        logarithmic_gate(q32)
    q64 = float64_expectations(cell.input_gate, combined)
    assert q64.dtype == torch.float64
    assert (q64.abs() < 1).all()


def test_cell_trains_through_boundary_expectations():
    model = build_quantum_model("qslstm_log", 1, 4, 1, 2, seed=0)
    _boundary_input_gate(model.cell)
    outputs, (h, c, n, scale) = model(torch.full((8, 5, 1), .3))
    assert outputs.dtype == h.dtype == c.dtype == n.dtype == torch.float32
    assert torch.isfinite(outputs).all() and (n > 0).all()
    outputs.sum().backward()
    for name, p in model.named_parameters():
        assert torch.isfinite(p.grad).all(), name
