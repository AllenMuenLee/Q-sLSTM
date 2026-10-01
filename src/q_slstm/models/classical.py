# Classical baselines with the quantum models' (outputs, state) return contract.
#
# ClassicalLSTM  : nn.LSTM + linear read-out.
# ClassicalSLSTM : xLSTM sLSTM with exponential input AND forget gates, normalizer n and log-domain
#                  stabilizer m, h = o * c / n. The same recurrence as CustomQsLSTMCell (stabilize_gates,
#                  normalized read-out) with one linear layer over [x, h] in place of the four VQCs, so
#                  exp(pre) plays the role of the amplified (1 + q) / (1 - q) gates.

import torch
from torch import nn

from .q_slstm_cell import DEFAULT_GATE_EPSILON, effective_epsilon, stabilize_gates

CLASSICAL_SLSTM_RECURRENCE = "classical_xlstm_exp_gates_v1"


class ClassicalLSTM(nn.Module):
    """nn.LSTM + linear read-out."""

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h, state = self.lstm(x)
        return self.head(h), state


class ClassicalSLSTM(nn.Module):
    """Classical sLSTM, i = exp(i~), f = exp(f~), state (h, c, n, m)."""

    recurrence = CLASSICAL_SLSTM_RECURRENCE

    def __init__(self, input_size, hidden_size, output_size, gate_epsilon=DEFAULT_GATE_EPSILON):
        super().__init__()
        self.hidden_size = hidden_size
        self.gate_epsilon = gate_epsilon
        self.gates = nn.Linear(input_size + hidden_size, 4 * hidden_size)
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        batch, length, _ = x.shape
        if hidden is None:
            hidden = tuple(x.new_zeros(batch, self.hidden_size) for _ in range(4))
        h, c, n, m = hidden
        n_floor = effective_epsilon(self.gate_epsilon, x.dtype)
        outputs = []
        for t in range(length):
            pre_i, pre_f, pre_z, pre_o = self.gates(torch.cat((x[:, t], h), dim=-1)).chunk(4, dim=-1)
            # log i = pre_i and log f = pre_f: both gates exponential, stabilized as in Q-sLSTM.
            m, i_prime, f_prime = stabilize_gates(pre_i, pre_f, m)
            c = f_prime * c + i_prime * torch.tanh(pre_z)
            n = f_prime * n + i_prime
            h = torch.sigmoid(pre_o) * (c / torch.clamp_min(n, n_floor))
            outputs.append(h)
        return self.head(torch.stack(outputs, dim=1)), (h, c, n, m)
