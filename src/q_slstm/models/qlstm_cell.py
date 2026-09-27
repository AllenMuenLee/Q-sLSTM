# Conventional QLSTM cell (baseline): four independent VQC gates with the classical
# sigmoid/tanh LSTM recurrence and (h, c) state. Same VQCs and output layer as the Q-sLSTM
# cell, so the two differ only in the gate transforms and recurrent update.

import torch
import torch.nn as nn

from .diagnostics import write_proportion
from .vqc import VQC

torch.set_default_dtype(torch.float32)


class CustomQLSTMCell(nn.Module):
    """Conventional QLSTM cell with recurrent state (h, c)."""

    def __init__(self, input_size, hidden_size, output_size, vqc_depth):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_qubits = input_size + hidden_size

        # Same construction order as CustomQsLSTMCell, so equal seeds give equal initial parameters.
        self.input_gate = VQC(vqc_depth=vqc_depth, n_qubits=self.n_qubits, n_class=hidden_size)
        self.forget_gate = VQC(vqc_depth=vqc_depth, n_qubits=self.n_qubits, n_class=hidden_size)
        self.cell_gate = VQC(vqc_depth=vqc_depth, n_qubits=self.n_qubits, n_class=hidden_size)
        self.output_gate = VQC(vqc_depth=vqc_depth, n_qubits=self.n_qubits, n_class=hidden_size)

        self.output_post_processing = nn.Linear(hidden_size, output_size)

    supports_alpha_diagnostics = True

    def forward(self, x, hidden, return_diagnostics=False, n_diag=None):
        """Conventional step. `return_diagnostics` adds an analysis-only write proportion.

        The baseline has no normalizer, so the diagnostic keeps a separate accumulator
        n_diag_after = f_t * n_diag_before + i_t (zero-initialized when `n_diag` is None) built from
        detached sigmoid gates. It never influences the state, output, loss, or gradients.
        """
        h_prev, c_prev = hidden

        # Concatenate input and hidden state
        combined = torch.cat((x, h_prev), dim=-1)

        i_t = torch.sigmoid(self.input_gate(combined))   # Input gate
        f_t = torch.sigmoid(self.forget_gate(combined))  # Forget gate
        g_t = torch.tanh(self.cell_gate(combined))       # Cell gate
        o_t = torch.sigmoid(self.output_gate(combined))  # Output gate

        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)

        out = self.output_post_processing(h_t)

        if return_diagnostics:
            n_before = torch.zeros_like(c_t) if n_diag is None else n_diag
            alpha, n_after = write_proportion(i_t, f_t, n_before)
            return out, h_t, c_t, {"alpha": alpha, "n_diag": n_after}
        return out, h_t, c_t
