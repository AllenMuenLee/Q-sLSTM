"""Q-sLSTM variant using ln(2/(1-q)) as the input and forget gate values."""

import math

import torch

from .q_slstm_cell import (
    DEFAULT_GATE_EPSILON,
    CustomQsLSTMCell,
    binary_scaled_memory_update,
)

QSLSTM_LOG_RECURRENCE = "log_gate_binary_scale_v1"


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


def logarithmic_memory_update(q_i, q_f, z_t, c_prev, n_prev, scale_prev, *, return_forget_weight=False):
    """C'=f*C+i*z, N'=f*N+i with i,f=ln(2/(1-q)) and binary scaling."""
    i = logarithmic_gate(q_i)
    f = logarithmic_gate(q_f)
    return binary_scaled_memory_update(
        f, i, torch.ones_like(i), z_t, c_prev, n_prev, scale_prev,
        return_forget_weight=return_forget_weight,
    )


class CustomQsLSTMLogCell(CustomQsLSTMCell):
    """Logarithmic gates with the same VQCs and (h,c,n,binary_scale) contract.

    gate_epsilon is accepted for API compatibility only; gates are not clipped.
    The VQCs and output projection are inherited; the memory update is its own.
    """

    memory_update = staticmethod(logarithmic_memory_update)

    def __init__(self, input_size, hidden_size, output_size, vqc_depth, gate_epsilon=DEFAULT_GATE_EPSILON):
        super().__init__(input_size, hidden_size, output_size, vqc_depth, gate_epsilon)
        self.recurrence = QSLSTM_LOG_RECURRENCE

    def forward(self, x, hidden, return_diagnostics=False):
        h_prev, c_prev, n_prev, scale_prev = hidden

        combined = torch.cat((x, h_prev), dim=-1)

        q_i = self.input_gate(combined)
        q_f = self.forget_gate(combined)
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
