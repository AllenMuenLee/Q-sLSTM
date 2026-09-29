# Reference ("fk") QLSTM cell, ported from src/fk_q_slstm/MQLSTM_32724.py.
#
# The conventional QLSTM recurrence with the reference's classical encoders around every gate VQC
#     Linear(input + hidden -> n_qubits) -> VQC -> Linear(n_qubits -> hidden)
#     i = sigmoid(.), f = sigmoid(.), g = tanh(.), o = sigmoid(.)
#     c_t = f * c_prev + i * g,  h_t = o * tanh(c_t)
# The reference circuit (AngleEmbedding + BasicEntanglerLayers) is replaced by the shared VQC used by
# qlstm / qslstm / qslstm_log. fk_qslstm (fk_q_slstm_cell.py) is this cell with exp input/forget gates.

import torch
import torch.nn as nn

from .diagnostics import write_proportion
from .vqc import VQC

torch.set_default_dtype(torch.float32)

FK_QLSTM_RECURRENCE = "fk_sigmoid_gates_classical_encoders_v1"


class CustomFkQLSTMCell(nn.Module):
    """Reference QLSTM cell with classical encoders around each gate VQC; recurrent state (h, c)."""

    supports_alpha_diagnostics = True
    input_activation = forget_activation = staticmethod(torch.sigmoid)
    recurrence = FK_QLSTM_RECURRENCE

    def __init__(self, input_size, hidden_size, output_size, vqc_depth):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_qubits = input_size + hidden_size

        # VQCs and output layer first, in the same order as the other cells, so equal seeds give the
        # same initial VQC and output parameters. Every qubit is measured; Elayer_out maps to hidden.
        self.input_gate = VQC(vqc_depth=vqc_depth, n_qubits=self.n_qubits, n_class=self.n_qubits)
        self.forget_gate = VQC(vqc_depth=vqc_depth, n_qubits=self.n_qubits, n_class=self.n_qubits)
        self.cell_gate = VQC(vqc_depth=vqc_depth, n_qubits=self.n_qubits, n_class=self.n_qubits)
        self.output_gate = VQC(vqc_depth=vqc_depth, n_qubits=self.n_qubits, n_class=self.n_qubits)

        self.output_post_processing = nn.Linear(hidden_size, output_size)

        # Classical encoders of the reference (four_Elayer_before_vqc=True, combine_Elayer_after_vqc=False).
        concat_size = input_size + hidden_size
        self.Elayer_in_input = nn.Linear(concat_size, self.n_qubits)
        self.Elayer_in_forget = nn.Linear(concat_size, self.n_qubits)
        self.Elayer_in_update = nn.Linear(concat_size, self.n_qubits)
        self.Elayer_in_output = nn.Linear(concat_size, self.n_qubits)
        self.Elayer_out_input = nn.Linear(self.n_qubits, hidden_size)
        self.Elayer_out_forget = nn.Linear(self.n_qubits, hidden_size)
        self.Elayer_out_update = nn.Linear(self.n_qubits, hidden_size)
        self.Elayer_out_output = nn.Linear(self.n_qubits, hidden_size)

    def gate_values(self, combined):
        """(i, f, g, o) for the concatenated [x, h_prev]."""
        encode = lambda vqc, name: getattr(self, f"Elayer_out_{name}")(vqc(getattr(self, f"Elayer_in_{name}")(combined)))
        return (self.input_activation(encode(self.input_gate, "input")),
                self.forget_activation(encode(self.forget_gate, "forget")),
                torch.tanh(encode(self.cell_gate, "update")),
                torch.sigmoid(encode(self.output_gate, "output")))

    def forward(self, x, hidden, return_diagnostics=False, n_diag=None):
        """One step. `return_diagnostics` adds an analysis-only write proportion.

        Like the QLSTM baseline, the cell has no normalizer, so the diagnostic keeps a separate
        accumulator n_diag_after = f_t * n_diag_before + i_t (zero-initialized when `n_diag` is None)
        from detached gates. It never influences the state, output, loss, or gradients.
        """
        h_prev, c_prev = hidden

        # Concatenate input and hidden state
        combined = torch.cat((x, h_prev), dim=-1)
        i_t, f_t, g_t, o_t = self.gate_values(combined)

        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)

        out = self.output_post_processing(h_t)

        if return_diagnostics:
            n_before = torch.zeros_like(c_t) if n_diag is None else n_diag
            alpha, n_after = write_proportion(i_t, f_t, n_before)
            return out, h_t, c_t, {"alpha": alpha, "n_diag": n_after}
        return out, h_t, c_t
