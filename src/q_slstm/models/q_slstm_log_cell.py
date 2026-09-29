"""Q-sLSTM variant using ln(2/(1-q)) as both the input and the forget gate."""

import math

import torch

from .q_slstm_cell import (
    DEFAULT_GATE_EPSILON,
    CustomQsLSTMCell,
    binary_scaled_memory_update,
)
from .vqc import VQC

# i = f = ln(2/(1-q)), evaluated on float64 VQC expectations, then cast to the state dtype.
QSLSTM_LOG_RECURRENCE = "log_gate_binary_scale_f64_gates_v2"
# Same amplified forget gate on float32 expectations: complex64 simulation overshoots |q| <= 1 and the
# float32 cast rounds q within ~3e-8 of +-1 onto +-1, so training could crash at q=1 or q=-1 (N=0).
QSLSTM_LOG_AMPLIFIED_FORGET_RECURRENCE = "log_gate_binary_scale_v1"
# Runs whose forget gate was sigmoid(q_f) (2026-09-27 .. 2026-09-29; kept for tracing those runs):
# float64 gates, and the float32 gates that crashed as described above.
QSLSTM_LOG_SIGMOID_FORGET_F64_RECURRENCE = "log_input_sigmoid_forget_binary_scale_f64_gates_v2"
QSLSTM_LOG_FLOAT32_GATES_RECURRENCE = "log_input_sigmoid_forget_binary_scale_v1"
QSLSTM_LOG_SIGMOID_FORGET_RECURRENCES = (QSLSTM_LOG_SIGMOID_FORGET_F64_RECURRENCE, QSLSTM_LOG_FLOAT32_GATES_RECURRENCE)


def float64_expectations(gate, X):
    """The gate's expectations [batch, n_class] simulated and returned in float64.

    VQC.forward simulates at its inputs' precision (float32 gives a complex64 statevector whose
    |<Z>| can exceed 1 by ~3e-7) and casts back to X.dtype, rounding q within ~3e-8 of +-1 onto +-1.
    Here the circuit runs on float64 inputs and weights; gradients still reach the float32 weights.
    Gates without a `VQC` circuit (torch stand-ins in tests) are evaluated on float64 inputs.
    """
    if isinstance(gate, VQC):
        expvals = torch.stack(gate.VQC(X.double(), gate.weights.double(), gate.n_class)).movedim(0, -1)
        if expvals.shape != (X.shape[0], gate.n_class):
            raise ValueError(f"VQC expected output [batch, {gate.n_class}], got {tuple(expvals.shape)}")
        return expvals
    return gate(X.double())


def logarithmic_gate(q):
    """Evaluate ln(2/(1-q)) on [-1, 1), without clipping or exponentiating.

    log1p preserves tiny positive gates near -1. The other branch avoids
    rounding (1+q)/2 to one near +1. q=-1 gives zero; q=1 is singular.
    """
    if not torch.isfinite(q).all():
        raise ValueError("VQC expectation values contain NaN or infinity")
    if ((q < -1) | (q > 1)).any():
        raise ValueError("VQC expectation outside [-1, 1]; clipping is disabled")
    if (q == 1).any():
        raise ValueError("VQC expectation q=1: singular logarithmic gate; clipping is disabled")

    # Keep both evaluated branches finite, including their backward operations.
    lower = -torch.log1p(-(1 + q.clamp(max=0)) / 2)
    upper = math.log(2) - torch.log1p(-q.clamp(min=0))
    return torch.where(q <= 0, lower, upper)


def sigmoid_forget_gate(q):
    """sigmoid(q) for a finite VQC expectation."""
    if not torch.isfinite(q).all():
        raise ValueError("VQC expectation values contain NaN or infinity")
    return torch.sigmoid(q)


def logarithmic_memory_update(q_i, q_f, z_t, c_prev, n_prev, scale_prev, *, return_forget_weight=False,
                              amplified_forget=True):
    """C'=f*C+i*z, N'=f*N+i with i=f=ln(2/(1-q)) (i from q_i, f from q_f), and binary scaling.

    `amplified_forget=False` uses f=sigmoid(q_f) instead, as in runs tagged
    QSLSTM_LOG_SIGMOID_FORGET_RECURRENCES.
    """
    # Gates are evaluated in the expectations' dtype (float64 from the cell) and only then cast to
    # the state dtype: cast first, q within ~3e-8 of +-1 would become exactly +-1.
    i = logarithmic_gate(q_i).to(z_t.dtype)
    f = (logarithmic_gate(q_f) if amplified_forget else sigmoid_forget_gate(q_f)).to(z_t.dtype)
    return binary_scaled_memory_update(
        f, i, torch.ones_like(i), z_t, c_prev, n_prev, scale_prev,
        return_forget_weight=return_forget_weight,
    )


class CustomQsLSTMLogCell(CustomQsLSTMCell):
    """Logarithmic input gate, sigmoid forget gate, same VQCs and (h,c,n,binary_scale) contract.

    gate_epsilon is accepted for API compatibility only; gates are not clipped.
    The VQCs and output projection are inherited; the memory update is its own.
    """

    memory_update = staticmethod(logarithmic_memory_update)

    def __init__(self, input_size, hidden_size, output_size, vqc_depth, gate_epsilon=DEFAULT_GATE_EPSILON):
        super().__init__(input_size, hidden_size, output_size, vqc_depth, gate_epsilon)
        self.recurrence = QSLSTM_LOG_RECURRENCE

    def gate_expectations(self, combined):
        """Float64 input/forget expectations; the log gate is singular at q=1, so no float32 rounding."""
        return float64_expectations(self.input_gate, combined), float64_expectations(self.forget_gate, combined)

    def forward(self, x, hidden, return_diagnostics=False):
        h_prev, c_prev, n_prev, scale_prev = hidden

        combined = torch.cat((x, h_prev), dim=-1)

        q_i, q_f = self.gate_expectations(combined)
        z_t = torch.tanh(self.cell_gate(combined))
        o_t = torch.sigmoid(self.output_gate(combined))

        c_t, n_t, scale_t, i_weight = self.memory_update(
            q_i, q_f, z_t, c_prev, n_prev, scale_prev)
        h_t = o_t * (c_t / n_t)

        y_t = self.output_post_processing(h_t)

        if return_diagnostics:
            alpha = i_weight.detach() / n_t.detach()
            return y_t, h_t, c_t, n_t, scale_t, {"alpha": alpha}
        return y_t, h_t, c_t, n_t, scale_t
