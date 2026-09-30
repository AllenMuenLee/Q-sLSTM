"""qslstm_sqrt: qslstm with sqrt((1+q)/(1-q)) input and forget gates."""

import torch

from q_slstm.experiments.nearest_neighbor import resolve_config as nn_config
from q_slstm.experiments.scalar_tasks import MODELS, resolve_config
from q_slstm.models.factory import QUANTUM_MODELS, build_quantum_model, recurrence_tag
from q_slstm.models.q_slstm_cell import stabilize_gates
from q_slstm.models.q_slstm_sqrt_cell import QSLSTM_SQRT_RECURRENCE, CustomQsLSTMSqrtCell


def test_gates_are_square_roots_of_the_ratio():
    cell = build_quantum_model("qslstm_sqrt", 1, 2, 1, 1, seed=0).cell
    q = torch.tensor([-.9, -.5, 0., .5, .9], dtype=torch.float64)
    ratio = (1 + q) / (1 - q)
    torch.testing.assert_close(cell.log_input(q), torch.log(ratio.sqrt()))
    torch.testing.assert_close(cell.log_forget(q), torch.log(ratio.sqrt()))


def test_forward_matches_direct_recurrence_and_shares_init_with_qslstm():
    sqrt_model = build_quantum_model("qslstm_sqrt", 1, 2, 1, 1, seed=3)
    base_model = build_quantum_model("qslstm", 1, 2, 1, 1, seed=3)
    assert isinstance(sqrt_model.cell, CustomQsLSTMSqrtCell)
    sa, sb = sqrt_model.state_dict(), base_model.state_dict()
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)

    cell = sqrt_model.cell
    x = torch.randn(4, 1)
    h, c, n, m = torch.zeros(4, 2), torch.zeros(4, 2), torch.zeros(4, 2), torch.zeros(4, 2)
    with torch.no_grad():
        y, h_t, c_t, n_t, m_t = cell(x, (h, c, n, m))
        combined = torch.cat((x, h), dim=-1)
        q_i, q_f, q_z, q_o = (g(combined) for g in (cell.input_gate, cell.forget_gate, cell.cell_gate, cell.output_gate))
        ell_i = 0.5 * (torch.log1p(q_i) - torch.log1p(-q_i))
        ell_f = 0.5 * (torch.log1p(q_f) - torch.log1p(-q_f))
        m_ref, i_p, f_p = stabilize_gates(ell_i, ell_f, m)
        c_ref = f_p * c + i_p * torch.tanh(q_z)
        n_ref = f_p * n + i_p
        h_ref = torch.sigmoid(q_o) * c_ref / n_ref
    for actual, expected in ((m_t, m_ref), (c_t, c_ref), (n_t, n_ref), (h_t, h_ref)):
        torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(y, cell.output_post_processing(h_ref))


def test_gradients_reach_every_vqc():
    model = build_quantum_model("qslstm_sqrt", 1, 2, 1, 1, seed=0)
    model(torch.randn(2, 3, 1))[0].sum().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name


def test_available_in_every_experiment_with_its_recurrence_tag():
    assert "qslstm_sqrt" in QUANTUM_MODELS and "qslstm_sqrt" in MODELS
    assert recurrence_tag("qslstm_sqrt") == QSLSTM_SQRT_RECURRENCE
    config = resolve_config({"model": "qslstm_sqrt", "task": "ema", "seed": 1, "scale": "pilot"})
    assert config["qslstm_recurrence"] == QSLSTM_SQRT_RECURRENCE
    assert nn_config({"model": "qslstm_sqrt", "scale": "pilot", "seed": 0})["qslstm_recurrence"] == QSLSTM_SQRT_RECURRENCE
