# Sequence wrappers that unroll a single cell over time.
#
# CustomLSTM  : classical two-state (h, c) contract, used by the LSTM baseline (unchanged).
# CustomQsLSTM : Q-sLSTM four-state (h, c, n, m) contract.
# The validation and unrolling helpers below are used by CustomQsLSTM only.
# Both wrappers accept `return_diagnostics=True` to also return the analysis-only write proportion.

import torch
import torch.nn as nn

torch.set_default_dtype(torch.float32)


def _check_sequence(x, input_size):
    if x.ndim != 3:
        raise ValueError(f"Expected input [batch, sequence, input_size], got shape {tuple(x.shape)}")
    if x.shape[1] == 0:
        raise ValueError("Sequence length must be at least 1")
    if x.shape[2] != input_size:
        raise ValueError(f"Expected input_size={input_size}, got last dimension {x.shape[2]}")


def _check_state(state, n_states, batch_size, hidden_size):
    if not isinstance(state, (tuple, list)) or len(state) != n_states:
        raise ValueError(f"Expected an initial state of {n_states} tensors")
    expected = (batch_size, hidden_size)
    for tensor in state:
        if tuple(tensor.shape) != expected:
            raise ValueError(f"Expected state tensors of shape {expected}, got {tuple(tensor.shape)}")


def _unroll(cell, x, state):
    """Run `cell(x_t, state) -> (y_t, *new_state)` over the time axis of batch-first `x`."""
    outputs = []
    for t in range(x.shape[1]):
        y_t, *state = cell(x[:, t, :], tuple(state))
        outputs.append(y_t)
    return torch.stack(outputs, dim=1), tuple(state)


def _unroll_with_alpha(cell, x, state):
    """Like `_unroll`, also collecting the cell's per-step write proportion [batch, seq, hidden]."""
    outputs, alphas = [], []
    for t in range(x.shape[1]):
        y_t, *state, diag = cell(x[:, t, :], tuple(state), return_diagnostics=True)
        outputs.append(y_t)
        alphas.append(diag["alpha"])
    return torch.stack(outputs, dim=1), tuple(state), torch.stack(alphas, dim=1)


class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, lstm_cell_QT):
        super(CustomLSTM, self).__init__()
        self.hidden_size = hidden_size

        # Single LSTM cell
        self.cell = lstm_cell_QT

    def forward(self, x, hidden=None, return_diagnostics=False, n_diag_init=None):
        if return_diagnostics:
            return self._forward_with_diagnostics(x, hidden, n_diag_init)
        batch_size, seq_len, _ = x.size()

        # Initialize hidden and cell states if not provided
        device = x.device

        if hidden is None:
            h_t = torch.zeros(batch_size, self.hidden_size, device=device)
            c_t = torch.zeros(batch_size, self.hidden_size, device=device)
        else:
            h_t, c_t = hidden

        outputs = []

        # Process sequence one time step at a time
        for t in range(seq_len):
            x_t = x[:, t, :]  # Extract the t-th time step
            out, h_t, c_t = self.cell(x_t, (h_t, c_t))  # Update hidden and cell states
            outputs.append(out.unsqueeze(1))  # Collect output for this time step

        outputs = torch.cat(outputs, dim=1)  # Concatenate outputs across all time steps
        return outputs, (h_t, c_t)

    def _forward_with_diagnostics(self, x, hidden, n_diag_init):
        """Same recurrence, plus per-step write proportions {"alpha": [batch, seq, hidden]}.

        `n_diag_init` only seeds the analysis accumulator; it cannot change outputs or gradients.
        """
        if not getattr(self.cell, "supports_alpha_diagnostics", False):
            raise ValueError("The wrapped cell does not provide write-proportion diagnostics")
        batch_size, seq_len, _ = x.size()
        zeros = x.new_zeros(batch_size, self.hidden_size)
        h_t, c_t = (zeros, zeros) if hidden is None else hidden
        n_diag = zeros if n_diag_init is None else n_diag_init

        outputs, alphas = [], []
        for t in range(seq_len):
            out, h_t, c_t, diag = self.cell(x[:, t, :], (h_t, c_t), return_diagnostics=True, n_diag=n_diag)
            n_diag = diag["n_diag"]
            outputs.append(out.unsqueeze(1))
            alphas.append(diag["alpha"])
        return torch.cat(outputs, dim=1), (h_t, c_t), {"alpha": torch.stack(alphas, dim=1)}


class CustomQsLSTM(nn.Module):
    """Four-state (h, c, n, m) sequence wrapper for the Q-sLSTM cell."""

    N_STATES = 4

    def __init__(self, input_size, hidden_size, qlstm_cell):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Single Q-sLSTM cell
        self.cell = qlstm_cell

    def forward(self, x, hidden=None, return_diagnostics=False):
        _check_sequence(x, self.input_size)
        batch_size = x.shape[0]

        if hidden is None:
            hidden = tuple(x.new_zeros(batch_size, self.hidden_size) for _ in range(self.N_STATES))
        else:
            _check_state(hidden, self.N_STATES, batch_size, self.hidden_size)

        if return_diagnostics:
            outputs, state, alpha = _unroll_with_alpha(self.cell, x, hidden)
            return outputs, state, {"alpha": alpha}

        outputs, (h_t, c_t, n_t, m_t) = _unroll(self.cell, x, hidden)
        return outputs, (h_t, c_t, n_t, m_t)
