# Quantum sequence model with a trainable input projection: Linear(raw_input_size, projection_size)
# followed by tanh at every timestep, then the production QLSTM / Q-sLSTM from `build_quantum_model`
# with input_size = projection_size. All raw feature channels reach the model; only the circuit input
# width is reduced (qubits per VQC = projection_size + hidden_size).

import torch
import torch.nn as nn

from .factory import build_quantum_model
from .q_slstm_cell import DEFAULT_GATE_EPSILON

PROJECTION_ACTIVATION = "tanh"


class ProjectedQuantumModel(nn.Module):
    def __init__(self, projection, core, raw_input_size, projection_size):
        super().__init__()
        self.projection = projection
        self.core = core
        self.raw_input_size = raw_input_size
        self.projection_size = projection_size
        self.n_qubits = core.cell.n_qubits

    def forward(self, x, hidden=None, return_diagnostics=False):
        if x.ndim != 3 or x.shape[-1] != self.raw_input_size:
            raise ValueError(f"expected input [batch, sequence, {self.raw_input_size}], got {tuple(x.shape)}")
        z = torch.tanh(self.projection(x))
        if return_diagnostics:
            return self.core(z, hidden, return_diagnostics=True)
        return self.core(z, hidden)


def build_projected_quantum_model(model, raw_input_size, projection_size, hidden_size, output_size, qnn_depth,
                                  gate_epsilon=DEFAULT_GATE_EPSILON, device="cpu", model_seed=None,
                                  projection_seed=None):
    """`qlstm` or `qslstm` behind a Linear+tanh projection.

    The core is built by `build_quantum_model(seed=model_seed)` and the projection under its own
    isolated RNG context seeded by `projection_seed`, so paired models get identical initial tensors
    for equal seeds without touching global RNG state.
    """
    core = build_quantum_model(model, projection_size, hidden_size, output_size, qnn_depth,
                               gate_epsilon=gate_epsilon, device="cpu", seed=model_seed)
    if projection_seed is None:
        projection = nn.Linear(raw_input_size, projection_size)
    else:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(projection_seed)
            projection = nn.Linear(raw_input_size, projection_size)
    return ProjectedQuantumModel(projection.float(), core, raw_input_size, projection_size).to(device)
