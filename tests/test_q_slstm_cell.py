import math

import pytest
import torch

from q_slstm.models.q_slstm_cell import stabilize_gates

from qslstm_test_utils import FixedGate
from qslstm_test_utils import make_cell


def t(values, dtype=torch.float64):
    return torch.tensor(values, dtype=dtype)


# ---- stabilized gates ------------------------------------------------------------

@pytest.mark.parametrize(
    "ell_i, ell_f, m_prev, expected_m, expected_i, expected_f",
    [
        # input branch wins
        ([2.0], [-1.0], [0.5], 2.0, 1.0, math.exp(-1.0 + 0.5 - 2.0)),
        # forget branch wins
        ([-3.0], [1.0], [2.0], 3.0, math.exp(-3.0 - 3.0), 1.0),
        # tie
        ([1.0], [0.25], [0.75], 1.0, 1.0, 1.0),
    ],
)
def test_stabilized_gates_branches(ell_i, ell_f, m_prev, expected_m, expected_i, expected_f):
    m_t, i_prime, f_prime = stabilize_gates(t(ell_i), t(ell_f), t(m_prev))
    assert m_t.item() == pytest.approx(expected_m)
    assert i_prime.item() == pytest.approx(expected_i)
    assert f_prime.item() == pytest.approx(expected_f)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_stabilized_gates_are_finite_and_at_most_one(dtype):
    generator = torch.Generator().manual_seed(0)
    ell_i = torch.randn(64, 7, generator=generator).mul(8).clamp(-14.5, 14.5).to(dtype)
    ell_f = torch.randn(64, 7, generator=generator).mul(8).clamp(-14.5, 14.5).to(dtype)
    m_prev = torch.randn(64, 7, generator=generator).mul(30).to(dtype)
    # force some exact ties
    ell_i[:4] = ell_f[:4] + m_prev[:4]

    m_t, i_prime, f_prime = stabilize_gates(ell_i, ell_f, m_prev)

    for gate in (i_prime, f_prime):
        assert torch.isfinite(gate).all()
        assert (gate >= 0).all() and (gate <= 1).all()
    assert torch.allclose(torch.maximum(i_prime, f_prime), torch.ones_like(i_prime))
    assert torch.isfinite(m_t).all()
    assert (m_t >= ell_i).all() and (m_t >= ell_f + m_prev).all()


# ---- deterministic single step ---------------------------------------------------

def test_single_step_matches_hand_calculation():
    hidden = 2
    q_i, q_f = [0.5, -0.4], [0.2, 0.6]
    q_z, q_o = [0.3, -0.7], [0.1, -0.2]
    gates = [FixedGate([q]) for q in (q_i, q_f, q_z, q_o)]
    cell = make_cell(input_size=1, hidden_size=hidden, output_size=1, gates=gates, dtype=torch.float64)

    x = t([[0.4], [-0.1]])
    h_prev = t([[0.1, -0.2], [0.3, 0.0]])
    c_prev = t([[0.5, -0.4], [0.2, 0.9]])
    n_prev = t([[1.0, 0.5], [0.25, 2.0]])
    m_prev = t([[0.7, -0.3], [1.5, 0.2]])

    y_t, h_t, c_t, n_t, m_t = cell(x, (h_prev, c_prev, n_prev, m_prev))

    # independent scalar reference
    def log_ratio(q):
        return math.log((1 + q) / (1 - q))

    for b in range(2):
        for k in range(hidden):
            ell_i, ell_f = log_ratio(q_i[k]), log_ratio(q_f[k])
            m = max(ell_f + m_prev[b, k].item(), ell_i)
            i_p = math.exp(ell_i - m)
            f_p = math.exp(ell_f + m_prev[b, k].item() - m)
            c = f_p * c_prev[b, k].item() + i_p * math.tanh(q_z[k])
            n = f_p * n_prev[b, k].item() + i_p
            h = (1 / (1 + math.exp(-q_o[k]))) * (c / max(n, 1e-6))

            assert m_t[b, k].item() == pytest.approx(m)
            assert c_t[b, k].item() == pytest.approx(c)
            assert n_t[b, k].item() == pytest.approx(n)
            assert h_t[b, k].item() == pytest.approx(h)

            # the same scaled gates come out of the shared stabilizer helper
            _, i_check, f_check = stabilize_gates(t(ell_i), t(ell_f), m_prev[b, k])
            assert i_check.item() == pytest.approx(i_p)
            assert f_check.item() == pytest.approx(f_p)

    assert torch.allclose(y_t, cell.output_post_processing(h_t))
    assert y_t.dtype == torch.float64


def test_normalizer_denominator_uses_epsilon_floor():
    gates = [FixedGate([[0.0]]), FixedGate([[0.0]]), FixedGate([[0.9]]), FixedGate([[0.0]])]
    cell = make_cell(1, 1, gates=gates, gate_epsilon=0.5, dtype=torch.float64)
    zero = t([[0.0]])
    # i' = f' = 1 with m_prev = 0 (ell = 0); n_prev = -3 gives n_t = -2, below the floor
    _, h_t, c_t, n_t, _ = cell(t([[0.0]]), (zero, zero, t([[-3.0]]), zero))
    assert n_t.item() == pytest.approx(-2.0)
    assert h_t.item() == pytest.approx(0.5 * c_t.item() / 0.5)
