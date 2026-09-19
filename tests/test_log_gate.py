import math

import pytest
import torch

from q_slstm.models.q_slstm_cell import CustomQsLSTMCell
from q_slstm.models.q_slstm_cell import bounded_log_ratio
from q_slstm.models.q_slstm_cell import effective_epsilon


def log_limit(epsilon, dtype):
    """Documented bound log(2 - eps) - log(eps), plus the rounding slack of forming 1 - eps in `dtype`."""
    eps = effective_epsilon(epsilon, dtype)
    rounding_slack = 2 * torch.finfo(dtype).eps / eps
    return math.log(2 - eps) - math.log(eps) + rounding_slack


RAW_VALUES = [-1.0, 0.0, 1.0, 1.001, -1.001, 5.0, -5.0, 1 - 1e-7, -1 + 1e-7, 1 - 1e-6, -1 + 1e-6, 1 - 1e-3]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("epsilon", [1e-6, 1e-3, 1e-12])
def test_log_gate_is_finite_and_within_epsilon_bounds(dtype, epsilon):
    q = torch.tensor(RAW_VALUES, dtype=dtype)
    log_gate = bounded_log_ratio(q, epsilon)

    assert log_gate.dtype == dtype
    assert torch.isfinite(log_gate).all()
    assert log_gate.abs().max().item() <= log_limit(epsilon, dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_log_gate_is_zero_at_zero_and_antisymmetric(dtype):
    q = torch.tensor([0.0, 0.3, 0.9], dtype=dtype)
    assert bounded_log_ratio(q[:1]).abs().item() < 1e-7
    assert torch.allclose(bounded_log_ratio(-q), -bounded_log_ratio(q))


def test_log_gate_matches_direct_log_ratio_inside_interval():
    q = torch.tensor([-0.7, -0.1, 0.25, 0.8], dtype=torch.float64)
    expected = torch.log((1 + q) / (1 - q))
    assert torch.allclose(bounded_log_ratio(q), expected)


def test_effective_epsilon_respects_dtype_machine_epsilon():
    assert effective_epsilon(1e-12, torch.float32) == torch.finfo(torch.float32).eps
    assert effective_epsilon(1e-6, torch.float32) == 1e-6
    assert effective_epsilon(1e-6, torch.float64) == 1e-6


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_log_gate_gradient_is_finite_near_upper_boundary(dtype):
    epsilon = 1e-6
    q = torch.tensor([1 - 1.5 * epsilon, -1 + 1.5 * epsilon, 0.0, 0.5], dtype=dtype, requires_grad=True)
    bounded_log_ratio(q, epsilon).sum().backward()

    assert torch.isfinite(q.grad).all()
    assert (q.grad > 0).all()
    if dtype == torch.float64:
        expected = 1 / (1 + q.detach()) + 1 / (1 - q.detach())
        assert torch.allclose(q.grad, expected)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_log_gate_rejects_non_finite_input(bad):
    with pytest.raises(ValueError, match="NaN or infinity"):
        bounded_log_ratio(torch.tensor([0.1, bad]))


@pytest.mark.parametrize("bad_epsilon", [0.0, -1e-6, 1.0, 1.5, float("nan")])
def test_cell_rejects_invalid_epsilon(bad_epsilon):
    with pytest.raises(ValueError, match="gate_epsilon"):
        CustomQsLSTMCell(1, 2, 1, vqc_depth=1, gate_epsilon=bad_epsilon)
