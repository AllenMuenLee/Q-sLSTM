"""Shared helpers: deterministic stand-ins for the VQC gates and a cell builder."""

import torch
import torch.nn as nn

from q_slstm.models.q_slstm_cell import CustomQsLSTMCell
from q_slstm.models.qlstm_cell import CustomQLSTMCell


class FixedGate(nn.Module):
    """Returns a constant [batch, hidden] tensor, ignoring the input values."""

    def __init__(self, values):
        super().__init__()
        self.register_buffer("values", torch.as_tensor(values))

    def forward(self, X):
        return self.values.to(X.dtype).expand(X.shape[0], -1)


class SmoothGate(nn.Module):
    """tanh(X @ W): deterministic, differentiable, values strictly inside (-1, 1)."""

    def __init__(self, n_in, hidden, seed):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.weights = nn.Parameter(torch.randn(n_in, hidden, generator=generator) * 0.8)

    def forward(self, X):
        return torch.tanh(X @ self.weights.to(X.dtype))


def make_cell(input_size, hidden_size, output_size=1, gate_epsilon=1e-6, gates=None, dtype=torch.float32):
    """Real cell with the four VQC modules replaced by torch stubs (fast, no simulator)."""
    torch.manual_seed(0)
    cell = CustomQsLSTMCell(input_size, hidden_size, output_size, vqc_depth=1, gate_epsilon=gate_epsilon)
    n_in = input_size + hidden_size
    names = ("input_gate", "forget_gate", "cell_gate", "output_gate")
    if gates is None:
        gates = [SmoothGate(n_in, hidden_size, seed=10 + k) for k in range(4)]
    for name, gate in zip(names, gates):
        setattr(cell, name, gate)
    return cell.to(dtype)


def make_qlstm_cell(input_size, hidden_size, output_size=1, gates=None, dtype=torch.float32):
    """Conventional QLSTM cell with torch stubs in place of the four VQCs."""
    torch.manual_seed(0)
    cell = CustomQLSTMCell(input_size, hidden_size, output_size, vqc_depth=1)
    n_in = input_size + hidden_size
    names = ("input_gate", "forget_gate", "cell_gate", "output_gate")
    if gates is None:
        gates = [SmoothGate(n_in, hidden_size, seed=10 + k) for k in range(4)]
    for name, gate in zip(names, gates):
        setattr(cell, name, gate)
    return cell.to(dtype)
