# Q-sLSTM cell (stabilized): four independent VQC gates feeding the normalized, xLSTM-stabilized
# (h, c, n, m) recurrence.
#
# 2025 06 19: QLSTM with Pennylane Batch support
# 2024 11 24: Modern QLSTM version

import torch
import torch.nn as nn

from .diagnostics import write_proportion
from .vqc import VQC

torch.set_default_dtype(torch.float32)

DEFAULT_GATE_EPSILON = 1e-6
# The unclipped polynomial recurrence was reverted: c and n still depend on the direct
# input/forget gate values, so it did not remove the need to clip q.
QSLSTM_RECURRENCE = "xlstm_stabilized_v1"
# Tag of runs trained with the reverted polynomial recurrence (kept for tracing those runs).
QSLSTM_POLYNOMIAL_RECURRENCE = "polynomial_binary_scale_v2"


class _BinaryScale(torch.autograd.Function):
    """Power-of-two scaling with an explicitly floating-point backward operation.

    Some torch versions' ldexp backward evaluates 2**integer_exponent in integer
    arithmetic, giving zero gradients for negative shifts. Applying ldexp to the
    gradient itself avoids that issue and supports higher derivatives as well.
    """

    @staticmethod
    def forward(ctx, value, exponent):
        ctx.save_for_backward(exponent)
        return torch.ldexp(value, exponent)

    @staticmethod
    def backward(ctx, grad_output):
        (exponent,) = ctx.saved_tensors
        return _BinaryScale.apply(grad_output, exponent), None


def effective_epsilon(epsilon, dtype):
    """Configured epsilon, but never below the machine epsilon of `dtype`."""
    return max(epsilon, torch.finfo(dtype).eps)


def bounded_log_ratio(q, epsilon=DEFAULT_GATE_EPSILON):
    """Log of (1 + q) / (1 - q) for a raw Pauli-Z expectation `q`, kept in the log domain.

    The expectation is clamped to [-1 + eps, 1 - eps] first, so the result is finite and
    bounded by log(2 - eps) - log(eps) in magnitude. The ratio itself is never exponentiated.
    """
    if not torch.isfinite(q).all():
        raise ValueError("VQC expectation values contain NaN or infinity")

    eps = effective_epsilon(epsilon, q.dtype)
    q_safe = torch.clamp(q, -1.0 + eps, 1.0 - eps)
    return torch.log1p(q_safe) - torch.log1p(-q_safe)


def stabilize_gates(ell_i, ell_f, m_prev):
    """xLSTM exponential stabilizer over log input/forget gates.

    Returns (m_t, i_prime, f_prime); both scaled gates lie in [0, 1] and at least one
    of them equals one for every element.
    """
    m_t = torch.maximum(ell_f + m_prev, ell_i)
    # Exact arithmetic gives arguments <= 0; the clamp only removes rounding noise.
    i_prime = torch.exp(torch.clamp(ell_i - m_t, max=0.0))
    f_prime = torch.exp(torch.clamp(ell_f + m_prev - m_t, max=0.0))
    return m_t, i_prime, f_prime


def polynomial_memory_update(q_i, q_f, z_t, c_prev, n_prev, scale_prev, *, return_forget_weight=False):
    """Reverted unclipped rational-gate recurrence; not used by CustomQsLSTMCell.

    Kept only so runs tagged QSLSTM_POLYNOMIAL_RECURRENCE can still be traced.

    Cancelling a common (1-q_f)(1-q_i) from the user's squared polynomial gives
        a = (1+q_f)(1-q_i), b = (1+q_i)(1-q_f), d = (1-q_f)(1-q_i).
    For *unscaled* states the normalized update is (a*C + b*z)/(a*N + b),
    but the next unscaled states are (a*C+b*z)/d and (a*N+b)/d.
    Dropping d from the recurrent state would change later writes' weights.

    Stored c,n represent C=c*2**scale, N=n*2**scale. frexp/ldexp carry that
    common scale using an integer-valued exponent, without materializing 2**scale.
    Normalize incoming and outgoing state mantissas together. For a valid memory
    (|c|<=n, |z|<=1), every successful update has .5<=n<1 and |c|<=n.
    Nonzero gate mantissas receive only nonpositive shifts. No expectation clamp,
    log/exp gate transform, or epsilon denominator floor is used.
    q=+1 is singular in the original recurrence; invalid/undefined states raise.
    The fourth state is a binary exponent, NOT the legacy logarithmic m state.
    """
    for name, q in (("input", q_i), ("forget", q_f)):
        if not torch.isfinite(q).all():
            raise ValueError(f"{name} VQC expectation contains NaN or infinity")
        if ((q < -1) | (q > 1)).any():
            raise ValueError(f"{name} VQC expectation outside [-1, 1]; clipping is disabled")
        if (q == 1).any():
            raise ValueError(f"{name} VQC expectation q=1: singular rational gate; clipping is disabled")
    a = (1 + q_f) * (1 - q_i)
    b = (1 + q_i) * (1 - q_f)
    d = (1 - q_f) * (1 - q_i)
    return binary_scaled_memory_update(
        a, b, d, z_t, c_prev, n_prev, scale_prev,
        return_forget_weight=return_forget_weight,
    )


def binary_scaled_memory_update(a, b, d, z_t, c_prev, n_prev, scale_prev, *, return_forget_weight=False):
    """Shared state arithmetic for validated nonnegative gates f=a/d and i=b/d.

    Callers provide finite a,b >= 0 and d > 0. Stored c,n represent the raw
    memory multiplied by 2**(-scale); returned weights use the new state scale.
    """
    if not all(torch.isfinite(s).all() for s in (z_t, c_prev, n_prev, scale_prev)):
        raise ValueError("Memory state/candidate must be finite")
    if (z_t.abs() > 1).any():
        raise ValueError("Memory candidate must be in [-1, 1]")
    if (n_prev < 0).any():
        raise ValueError("Memory normalizer must be nonnegative")
    if ((n_prev == 0) & (c_prev != 0)).any():
        raise ValueError("An empty normalizer requires an empty cell state")
    # Range also guarantees the exponent is represented exactly in the state dtype.
    exponent_limit = min(2**30, int(1 / torch.finfo(scale_prev.dtype).eps))
    if ((scale_prev != torch.round(scale_prev)) | (scale_prev.abs() > exponent_limit)).any():
        raise ValueError("Expected a bounded integer binary scale, not a legacy log stabilizer state")

    # This also accepts previously valid but very large finite supplied states.
    # Normalize BEFORE multiplication so an intermediate cannot overflow first.
    _, previous_shift = torch.frexp(torch.maximum(c_prev.abs(), n_prev))
    previous_shift = previous_shift.to(torch.int64)
    c_prev = _BinaryScale.apply(c_prev, -previous_shift)
    n_prev = _BinaryScale.apply(n_prev, -previous_shift)
    previous_exponent = scale_prev.to(torch.int64) + previous_shift

    a_mantissa, a_exponent = torch.frexp(a)
    b_mantissa, b_exponent = torch.frexp(b)
    d_mantissa, d_exponent = torch.frexp(d)
    f_exponent = a_exponent.to(torch.int64) - d_exponent + previous_exponent
    i_exponent = b_exponent.to(torch.int64) - d_exponent
    # A zero branch has no meaningful exponent and must not suppress the other branch.
    next_exponent = torch.where(
        a == 0, i_exponent,
        torch.where(b == 0, f_exponent, torch.maximum(f_exponent, i_exponent)),
    )
    if (next_exponent.abs() > exponent_limit).any():
        raise ValueError("Memory binary scale exceeded its exact representation range")
    f_shift = f_exponent - next_exponent
    i_shift = i_exponent - next_exponent
    f_weight = _BinaryScale.apply(a_mantissa / d_mantissa, f_shift)
    i_weight = _BinaryScale.apply(b_mantissa / d_mantissa, i_shift)
    c_t = f_weight * c_prev + i_weight * z_t
    n_t = f_weight * n_prev + i_weight
    if not torch.isfinite(c_t).all() or not torch.isfinite(n_t).all():
        raise ValueError("Memory update produced a non-finite state")
    if (n_t <= 0).any():
        raise ValueError("Memory update has a zero denominator (no retained or new memory)")
    # Gate scaling alone is insufficient: repeated fractional mantissas can
    # still grow or shrink the stored states exponentially. Carry their common
    # exponent too. Positive normalizers from valid memory now lie in [.5,1).
    _, state_shift = torch.frexp(torch.maximum(c_t.abs(), n_t))
    state_shift = state_shift.to(torch.int64)
    c_t = _BinaryScale.apply(c_t, -state_shift)
    n_t = _BinaryScale.apply(n_t, -state_shift)
    i_weight = _BinaryScale.apply(i_weight, -state_shift)
    next_exponent = next_exponent + state_shift
    if (next_exponent.abs() > exponent_limit).any():
        raise ValueError("Memory binary scale exceeded its exact representation range")
    result = (c_t, n_t, next_exponent.to(scale_prev.dtype), i_weight)
    if return_forget_weight:
        # Trace weights multiply the original incoming stored states, before
        # this function's incoming-state normalization.
        f_weight = _BinaryScale.apply(f_weight, -previous_shift - state_shift)
        return (*result, f_weight)
    return result


class CustomQsLSTMCell(nn.Module):
    """Q-sLSTM cell with recurrent state (h, c, n, m)."""

    def __init__(self, input_size, hidden_size, output_size, vqc_depth, gate_epsilon=DEFAULT_GATE_EPSILON):
        super().__init__()
        if not 0.0 < gate_epsilon < 1.0:
            raise ValueError(f"gate_epsilon must satisfy 0 < epsilon < 1, got {gate_epsilon}")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_qubits = input_size + hidden_size
        self.gate_epsilon = gate_epsilon
        self.recurrence = QSLSTM_RECURRENCE

        # Four independently parameterized VQCs, each returning `hidden_size` expectations.
        self.input_gate = VQC(vqc_depth=vqc_depth, n_qubits=self.n_qubits, n_class=hidden_size)
        self.forget_gate = VQC(vqc_depth=vqc_depth, n_qubits=self.n_qubits, n_class=hidden_size)
        self.cell_gate = VQC(vqc_depth=vqc_depth, n_qubits=self.n_qubits, n_class=hidden_size)
        self.output_gate = VQC(vqc_depth=vqc_depth, n_qubits=self.n_qubits, n_class=hidden_size)

        self.output_post_processing = nn.Linear(hidden_size, output_size)

    supports_alpha_diagnostics = True

    def forward(self, x, hidden, return_diagnostics=False):
        h_prev, c_prev, n_prev, m_prev = hidden

        # Concatenate input and hidden state
        combined = torch.cat((x, h_prev), dim=-1)

        q_i = self.input_gate(combined)
        q_f = self.forget_gate(combined)
        q_z = self.cell_gate(combined)
        q_o = self.output_gate(combined)

        ell_i = bounded_log_ratio(q_i, self.gate_epsilon)
        ell_f = bounded_log_ratio(q_f, self.gate_epsilon)

        m_t, i_prime, f_prime = stabilize_gates(ell_i, ell_f, m_prev)

        z_t = torch.tanh(q_z)
        o_t = torch.sigmoid(q_o)

        c_t = f_prime * c_prev + i_prime * z_t
        n_t = f_prime * n_prev + i_prime

        n_floor = effective_epsilon(self.gate_epsilon, n_t.dtype)
        h_t = o_t * (c_t / torch.clamp_min(n_t, n_floor))

        y_t = self.output_post_processing(h_t)

        if return_diagnostics:
            # Analysis only: the actual stabilized gates and incoming normalizer, detached.
            alpha, _ = write_proportion(i_prime, f_prime, n_prev, self.gate_epsilon)
            return y_t, h_t, c_t, n_t, m_t, {"alpha": alpha}
        return y_t, h_t, c_t, n_t, m_t
