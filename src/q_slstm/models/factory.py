# Importable construction of quantum sequence models compared in the experiments.

import torch

from .q_slstm_cell import DEFAULT_GATE_EPSILON, CustomQsLSTMCell
from .q_slstm_log_cell import CustomQsLSTMLogCell
from .qlstm_cell import CustomQLSTMCell
from .sequence_wrappers import CustomLSTM, CustomQsLSTM

QUANTUM_MODELS = ("qlstm", "qslstm", "qslstm_log")


def build_quantum_model(model, input_size, hidden_size, output_size, qnn_depth,
                        gate_epsilon=DEFAULT_GATE_EPSILON, device="cpu", seed=None):
    """Build `qlstm`, `qslstm` (stabilized, (h, c, n, m)), or `qslstm_log` (ln(2/(1-q)) gates,
    (h, c, n, binary_scale)). All models construct their
    four VQCs and the output layer in the same order, so an equal
    `seed` gives equal initial parameters. Global RNG state is left untouched when `seed` is given.
    """
    if model not in QUANTUM_MODELS:
        raise ValueError(f"model must be one of {QUANTUM_MODELS}, got {model!r}")

    def build():
        if model == "qlstm":
            cell = CustomQLSTMCell(input_size, hidden_size, output_size, qnn_depth).float()
            return CustomLSTM(input_size, hidden_size, cell).float()
        cell_type = CustomQsLSTMLogCell if model == "qslstm_log" else CustomQsLSTMCell
        cell = cell_type(input_size, hidden_size, output_size, qnn_depth, gate_epsilon=gate_epsilon).float()
        return CustomQsLSTM(input_size, hidden_size, cell).float()

    if seed is None:
        net = build()
    else:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            net = build()
    return net.to(device)


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
