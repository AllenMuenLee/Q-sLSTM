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
