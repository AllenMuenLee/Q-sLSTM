"""Independent algebra, recurrent equivalence, and extreme-value regression checks."""

from decimal import Decimal, localcontext

import pytest
import torch

from q_slstm.models.q_slstm_cell import polynomial_memory_update


@pytest.mark.parametrize("qi,qf", [(-1., .2), (.2, -1.)])
def test_zero_gate_boundary_derivatives_match_unscaled_recurrence(qi, qf):
    qi = torch.tensor([qi], dtype=torch.float64, requires_grad=True)
    qf = torch.tensor([qf], dtype=torch.float64, requires_grad=True)
    z, c, n, scale = (torch.tensor([v], dtype=torch.float64) for v in (.7, .3, 2., 3.))
    cn, nn, _, _ = polynomial_memory_update(qi, qf, z, c, n, scale)
    i, f = (1 + qi) / (1 - qi), (1 + qf) / (1 - qf)
    reference = (f * c * 8 + i * z) / (f * n * 8 + i)
    got = torch.autograd.grad((cn / nn).sum(), (qi, qf))
    want = torch.autograd.grad(reference.sum(), (qi, qf))
    for left, right in zip(got, want):
        torch.testing.assert_close(left, right, rtol=1e-12, atol=1e-12)


def test_binary_shift_first_and_second_derivatives():
    from q_slstm.models.q_slstm_cell import _BinaryScale

    value = torch.tensor([.3, .8], dtype=torch.float64, requires_grad=True)
    shift = torch.tensor([-3, -1], dtype=torch.int64)
    output = _BinaryScale.apply(value, shift).square().sum()
    grad, = torch.autograd.grad(output, value, create_graph=True)
    second, = torch.autograd.grad(grad.sum(), value)
    factors = torch.tensor([1 / 8, 1 / 2], dtype=torch.float64)
    torch.testing.assert_close(grad, 2 * value * factors.square())
    torch.testing.assert_close(second, 2 * factors.square())


def test_matches_users_expanded_fraction_without_changing_gate_values():
    qi = torch.tensor([-.8, .2, .99999999, -.99999999], dtype=torch.float64)
    qf = torch.tensor([.4, -.5, .9999999, -.9999999], dtype=torch.float64)
    c, n, z = torch.tensor(.3), torch.tensor(1.5), torch.tensor(-.6)
    a = (1 - qf.square()) * (1 - qi).square()
    b = (1 - qi.square()) * (1 - qf).square()
    expected = (a * c + b * z) / (a * n + b)
    got_c, got_n, _, _ = polynomial_memory_update(qi, qf, z, c, n, torch.zeros_like(qi))
    # The expanded 1-q**2 itself loses precision near +/-1; the factored implementation does not.
    torch.testing.assert_close(got_c / got_n, expected, rtol=1e-8, atol=1e-9)


def test_multistep_outputs_and_gradients_match_unscaled_rational_recurrence():
    gen = torch.Generator().manual_seed(721)
    qi = (torch.rand(20, 3, generator=gen, dtype=torch.float64) * 1.8 - .9).requires_grad_()
    qf = (torch.rand(20, 3, generator=gen, dtype=torch.float64) * 1.8 - .9).requires_grad_()
    z = torch.randn(20, 3, generator=gen, dtype=torch.float64).tanh().requires_grad_()
    c = n = scale = torch.zeros(3, dtype=torch.float64)
    direct_c = direct_n = torch.zeros_like(c)
    actual, expected, actual_alpha, expected_alpha = [], [], [], []
    for t in range(len(qi)):
        i, f = (1 + qi[t]) / (1 - qi[t]), (1 + qf[t]) / (1 - qf[t])
        direct_c, direct_n = f * direct_c + i * z[t], f * direct_n + i
        c, n, scale, i_weight = polynomial_memory_update(qi[t], qf[t], z[t], c, n, scale)
        actual.append(c / n)
        expected.append(direct_c / direct_n)
        actual_alpha.append(i_weight / n)
        expected_alpha.append(i / direct_n)
    actual, expected = torch.stack(actual), torch.stack(expected)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(torch.stack(actual_alpha), torch.stack(expected_alpha))
    got_grad = torch.autograd.grad(actual.square().sum(), (qi, qf, z))
    want_grad = torch.autograd.grad(expected.square().sum(), (qi, qf, z))
    for got, want in zip(got_grad, want_grad):
        torch.testing.assert_close(got, want, rtol=1e-10, atol=1e-10)


def test_common_state_rescaling_preserves_prediction_and_next_unscaled_state():
    qi, qf, z = (torch.tensor([v], dtype=torch.float64) for v in (.7, -.4, .2))
    c, n = torch.tensor([.3], dtype=torch.float64), torch.tensor([.8], dtype=torch.float64)
    a = polynomial_memory_update(qi, qf, z, c, n, torch.zeros_like(c))
    b = polynomial_memory_update(qi, qf, z, c / 32, n / 32, torch.full_like(c, 5))
    torch.testing.assert_close(a[0] / a[1], b[0] / b[1])
    for index in (0, 1, 3):
        torch.testing.assert_close(
            torch.ldexp(a[index], a[2].to(torch.int64)),
            torch.ldexp(b[index], b[2].to(torch.int64)),
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_persistent_amplification_keeps_stored_states_bounded(dtype):
    # q_f=.2 means f=1.5: gate-exponent scaling alone leaves a repeated factor
    # of 1.5 in the stored state. The normalizer must be rescaled every step.
    qi, qf, z = (torch.tensor([v], dtype=dtype) for v in (.3, .2, .25))
    c = n = scale = torch.zeros(1, dtype=dtype)
    for _ in range(5000):
        c, n, scale, _ = polynomial_memory_update(qi, qf, z, c, n, scale)
        assert .5 <= n.item() < 1
        assert c.abs().item() <= n.item()
        torch.testing.assert_close(c / n, z)
    assert scale.item() > 1024  # raw N cannot be represented even in float64


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_large_finite_incoming_state_is_rescaled_before_multiplication(dtype):
    n = torch.tensor([torch.finfo(dtype).max * .75], dtype=dtype)
    c, scale = n / 4, torch.zeros_like(n)
    qi, qf, z = (torch.tensor([v], dtype=dtype) for v in (0., .5, .7))
    c, n, scale, _ = polynomial_memory_update(qi, qf, z, c, n, scale)
    assert .5 <= n.item() < 1
    assert c.abs().item() <= n.item()
    assert (c / n).item() == pytest.approx(.25)
    assert torch.isfinite(scale).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_first_tiny_write_is_not_blurred_by_epsilon(dtype):
    near_one = torch.nextafter(torch.ones(1, dtype=dtype), torch.zeros(1, dtype=dtype))
    qi, qf = -near_one, near_one
    z, zero = torch.full_like(qi, .75), torch.zeros_like(qi)
    c, n, scale, i = polynomial_memory_update(qi, qf, z, zero, zero, zero)
    assert torch.ldexp(n, scale.to(torch.int64)).item() < 1e-6
    torch.testing.assert_close(c / n, z)
    torch.testing.assert_close(i / n, torch.ones_like(i))


def test_extreme_accumulation_then_forgetting_matches_decimal_reference():
    # The raw normalizer exceeds float64 capacity; later forgetting must still recover new writes.
    with localcontext() as ctx:
        ctx.prec = 90
        dc = dn = Decimal(0)
        c = n = scale = torch.zeros(1, dtype=torch.float64)
        max_scale = 0
        for step in range(250):
            qf = .999 if step < 120 else -.999
            z = -.4 if step < 120 else .7
            qi_t, qf_t, z_t = (torch.tensor([v], dtype=torch.float64) for v in (.2, qf, z))
            qi_d, qf_d, z_d = (Decimal.from_float(v) for v in (.2, qf, z))
            i, f = (1 + qi_d) / (1 - qi_d), (1 + qf_d) / (1 - qf_d)
            dc, dn = f * dc + i * z_d, f * dn + i
            c, n, scale, _ = polynomial_memory_update(qi_t, qf_t, z_t, c, n, scale)
            max_scale = max(max_scale, scale.item())
            assert (c / n).item() == pytest.approx(float(dc / dn), abs=1e-12)
        assert max_scale > 1024
        assert (c / n).item() == pytest.approx(.7)


@pytest.mark.parametrize("qi,qf,match", [
    (1., .5, "singular"), (.5, 1., "singular"), (1., 1., "singular"),
    (-1., -1., "zero denominator"), (-1., .5, "zero denominator"),
    (1.01, .5, "outside"), (-1.01, .5, "outside"),
    (float("nan"), 0., "NaN"), (0., float("inf"), "infinity"),
])
def test_undefined_inputs_are_reported_instead_of_clipped(qi, qf, match):
    qi, qf = torch.tensor([qi]), torch.tensor([qf])
    zero = torch.zeros(1)
    with pytest.raises(ValueError, match=match):
        polynomial_memory_update(qi, qf, zero + .2, zero, zero, zero)


def test_zero_input_gate_retains_existing_memory_and_zero_forget_gate_replaces_it():
    c, n, scale = torch.tensor([.3]), torch.tensor([2.]), torch.zeros(1)
    got_c, got_n, _, i = polynomial_memory_update(
        torch.tensor([-1.]), torch.tensor([.2]), torch.tensor([.7]), c, n, scale)
    torch.testing.assert_close(got_c / got_n, c / n)
    assert i.item() == 0
    got_c, got_n, _, i = polynomial_memory_update(
        torch.tensor([.2]), torch.tensor([-1.]), torch.tensor([.7]), c, n, scale)
    torch.testing.assert_close(got_c / got_n, torch.tensor([.7]))
    torch.testing.assert_close(i / got_n, torch.ones_like(i))
