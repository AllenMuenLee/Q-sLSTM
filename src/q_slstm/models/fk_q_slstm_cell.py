# Reference ("fk") Q-sLSTM cell, ported from src/fk_q_slstm/QsLSTM_102124.py for the scalar tasks.
#
# Keeps the reference's classical encoders around every gate VQC
#     Linear(input + hidden -> n_qubits) -> VQC -> Linear(n_qubits -> hidden)
# and its sLSTMCell recurrence (exponential input/forget gates, no normalizer, no stabilizer):
#     i = exp(.), f = exp(.), g = tanh(.), o = sigmoid(.)
#     c_t = f * c_prev + i * g,  h_t = o * tanh(c_t)
# The reference circuit (AngleEmbedding + BasicEntanglerLayers) is replaced by the shared VQC used by
# qlstm / qslstm / qslstm_log. Without a stabilizer c_t can grow geometrically; a non-finite state is
# caught by the trainer's finiteness checks and recorded as a failed run.

import torch
import torch.nn as nn

from .vqc import VQC

torch.set_default_dtype(torch.float32)

FK_QSLSTM_RECURRENCE = "fk_exp_gates_classical_encoders_v1"


class CustomFkQsLSTMCell(nn.Module):
    """Reference Q-sLSTM cell with recurrent state (h, c)."""

    supports_alpha_diagnostics = False

    def __init__(self, input_size, hidden_size, output_size, vqc_depth):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_qubits = input_size + hidden_size
        self.recurrence = FK_QSLSTM_RECURRENCE

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

    def forward(self, x, hidden):
        h_prev, c_prev = hidden

        # Concatenate input and hidden state
        combined = torch.cat((x, h_prev), dim=-1)

        i_t = torch.exp(self.Elayer_out_input(self.input_gate(self.Elayer_in_input(combined))))
        f_t = torch.exp(self.Elayer_out_forget(self.forget_gate(self.Elayer_in_forget(combined))))
        g_t = torch.tanh(self.Elayer_out_update(self.cell_gate(self.Elayer_in_update(combined))))
        o_t = torch.sigmoid(self.Elayer_out_output(self.output_gate(self.Elayer_in_output(combined))))

        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)

        out = self.output_post_processing(h_t)
        return out, h_t, c_t
