"""Write-proportion diagnostics and QLSTM / Q-sLSTM parity for the nearest-neighbor experiment."""

import math

import numpy as np
import pytest
import torch

from q_slstm.models.diagnostics import write_proportion
from q_slstm.models.factory import build_quantum_model, count_trainable_parameters
from q_slstm.models.sequence_wrappers import CustomLSTM, CustomQsLSTM

from qslstm_test_utils import FixedGate, make_cell, make_qlstm_cell

IN, HID = 3, 2


def test_alpha_matches_hand_calculation_and_is_reduced_after_the_ratio():
    i = torch.tensor([[0.2, 0.5]])
    f = torch.tensor([[0.5, 0.25]])
    n_before = torch.tensor([[0.6, 2.0]])
    alpha, n_after = write_proportion(i, f, n_before)
    assert torch.allclose(n_after, torch.tensor([[0.5, 1.0]]))
    assert torch.allclose(alpha, torch.tensor([[0.4, 0.5]]))
    assert ((alpha >= 0) & (alpha <= 1)).all()
    # mean of elementwise ratios, not ratio of means
    assert alpha.mean().item() == pytest.approx(0.45)
    assert (i.mean() / (f * n_before + i).mean()).item() != pytest.approx(0.45)


def _bounded_log(q, eps=1e-6):
    q = min(max(q, -1 + eps), 1 - eps)
    return math.log1p(q) - math.log1p(-q)


def test_qslstm_diagnostics_use_actual_stabilized_gates_and_normalizer():
    qi, qf = [0.3, -0.2], [0.1, 0.5]
    gates = [FixedGate([qi]), FixedGate([qf]), FixedGate([[0.0, 0.0]]), FixedGate([[0.0, 0.0]])]
    cell = make_cell(IN, HID, 1, gates=gates)
    model = CustomQsLSTM(IN, HID, cell)
    x = torch.randn(1, 3, IN)
    _, _, diag = model(x, return_diagnostics=True)
    alpha = diag["alpha"][0].numpy()

    ell_i = np.array([_bounded_log(q) for q in qi])
    ell_f = np.array([_bounded_log(q) for q in qf])
    m_prev, n_prev = np.zeros(HID), np.zeros(HID)
    for t in range(3):
        m = np.maximum(ell_f + m_prev, ell_i)
        i_prime, f_prime = np.exp(ell_i - m), np.exp(ell_f + m_prev - m)
        n = f_prime * n_prev + i_prime
        np.testing.assert_allclose(alpha[t], i_prime / n, rtol=1e-5)
        m_prev, n_prev = m, n
    np.testing.assert_allclose(alpha[0], 1.0)  # empty memory: the first write is the whole memory


def test_qlstm_diagnostic_accumulator_cannot_change_predictions_or_gradients():
    def run(n_diag_init):
        cell = make_qlstm_cell(IN, HID, 1)
        model = CustomLSTM(IN, HID, cell)
        x = torch.randn(2, 4, IN, generator=torch.Generator().manual_seed(0))
        out, _, diag = model(x, return_diagnostics=True, n_diag_init=n_diag_init)
        out.pow(2).sum().backward()
        grads = torch.cat([p.grad.flatten() for p in model.parameters()])
        return out.detach(), grads, diag["alpha"]

    out_a, grad_a, alpha_a = run(None)
    out_b, grad_b, alpha_b = run(torch.full((2, HID), 5.0))
    assert torch.equal(out_a, out_b) and torch.equal(grad_a, grad_b)
    assert not torch.allclose(alpha_a[:, 0], alpha_b[:, 0])
    assert ((alpha_a >= 0) & (alpha_a <= 1)).all() and ((alpha_b >= 0) & (alpha_b <= 1)).all()
    assert not alpha_a.requires_grad  # analysis-only, detached

    # the diagnostic path leaves the ordinary forward untouched
    cell = make_qlstm_cell(IN, HID, 1)
    model = CustomLSTM(IN, HID, cell)
    x = torch.randn(2, 4, IN)
    plain, _ = model(x)
    diagnosed, _, _ = model(x, return_diagnostics=True)
    assert torch.equal(plain, diagnosed)


def test_qslstm_diagnostics_do_not_change_outputs():
    model = CustomQsLSTM(IN, HID, make_cell(IN, HID, 1))
    x = torch.randn(2, 4, IN)
    plain, state = model(x)
    diagnosed, state_d, diag = model(x, return_diagnostics=True)
    assert torch.equal(plain, diagnosed) and all(torch.equal(a, b) for a, b in zip(state, state_d))
    assert diag["alpha"].shape == (2, 4, HID)


def _build_pair(hidden=2, depth=1, seed=123):
    return (build_quantum_model("qlstm", IN, hidden, 1, depth, seed=seed),
            build_quantum_model("qslstm", IN, hidden, 1, depth, seed=seed))


def test_qlstm_and_qslstm_share_topology_and_paired_initial_parameters():
    torch_state = torch.get_rng_state()
    qlstm, qslstm = _build_pair()
    assert torch.equal(torch_state, torch.get_rng_state())  # global RNG untouched

    assert isinstance(qlstm, CustomLSTM) and isinstance(qslstm, CustomQsLSTM)
    for net in (qlstm, qslstm):
        cell = net.cell
        assert cell.n_qubits == IN + HID
        for gate in (cell.input_gate, cell.forget_gate, cell.cell_gate, cell.output_gate):
            assert gate.weights.shape == (1, IN + HID) and gate.n_class == HID and gate.n_qubits == IN + HID
        assert cell.output_post_processing.weight.shape == (1, HID)
    assert count_trainable_parameters(qlstm) == count_trainable_parameters(qslstm)

    sd_a, sd_b = qlstm.state_dict(), qslstm.state_dict()
    assert sd_a.keys() == sd_b.keys()
    assert all(torch.equal(sd_a[k], sd_b[k]) for k in sd_a)
    p_a, p_b = next(qlstm.parameters()), next(qslstm.parameters())
    assert p_a.data_ptr() != p_b.data_ptr()  # equal values, never tied
    with torch.no_grad():
        p_a.add_(1.0)
    assert not torch.equal(p_a, p_b)

    # a different seed gives different initial parameters
    other = build_quantum_model("qlstm", IN, HID, 1, 1, seed=124)
    assert not torch.equal(other.state_dict()["cell.input_gate.weights"], sd_b["cell.input_gate.weights"])


@pytest.mark.parametrize("name", ["qlstm", "qslstm"])
def test_wrappers_accept_task_inputs_and_backpropagate_finitely(name):
    model = build_quantum_model(name, IN, HID, 1, 1, seed=0)
    x = torch.rand(2, 4, IN)
    outputs = model(x)[0]
    assert outputs.shape == (2, 4, 1)
    loss = outputs.pow(2).mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    outputs, _, diag = model(x, return_diagnostics=True)
    assert diag["alpha"].shape == (2, 4, HID)
    assert torch.isfinite(diag["alpha"]).all() and (diag["alpha"] >= -1e-6).all() and (diag["alpha"] <= 1 + 1e-6).all()


def test_factory_rejects_unknown_model():
    with pytest.raises(ValueError):
        build_quantum_model("lstm", IN, HID, 1, 1)
