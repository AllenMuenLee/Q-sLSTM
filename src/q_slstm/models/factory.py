# Importable construction of quantum sequence models compared in the experiments.

import torch

from .fk_q_slstm_cell import CustomFkQsLSTMCell
from .fk_qlstm_cell import CustomFkQLSTMCell
from .q_slstm_cell import DEFAULT_GATE_EPSILON, CustomQsLSTMCell
from .q_slstm_log_cell import CustomQsLSTMLogCell
from .q_slstm_sqrt_cell import CustomQsLSTMSqrtCell
from .qlstm_cell import CustomQLSTMCell
from .sequence_wrappers import CustomLSTM, CustomQsLSTM

# Every experiment runs these with the same configuration (hidden size, depth, qubits, training);
# parameter counts are not matched (the fk_* models add classical encoders around their VQCs).
QUANTUM_MODELS = ("qlstm", "qslstm", "qslstm_log", "qslstm_sqrt", "fk_qslstm", "fk_qlstm")
QSLSTM_CELLS = {"qslstm": CustomQsLSTMCell, "qslstm_log": CustomQsLSTMLogCell, "qslstm_sqrt": CustomQsLSTMSqrtCell}
FK_CELLS = {"fk_qslstm": CustomFkQsLSTMCell, "fk_qlstm": CustomFkQLSTMCell}


def build_quantum_model(model, input_size, hidden_size, output_size, qnn_depth,
                        gate_epsilon=DEFAULT_GATE_EPSILON, device="cpu", seed=None):
    """Build `qlstm`, `qslstm` (stabilized, (h, c, n, m)), or `qslstm_log` (ln(2/(1-q)) gates,
    (h, c, n, binary_scale)), `qslstm_sqrt` (qslstm with sqrt((1+q)/(1-q)) input and forget gates,
    (h, c, n, m)), `fk_qslstm` (reference exp-gate cell with classical encoders, (h, c)), or
    `fk_qlstm` (the same encoders with sigmoid gates, (h, c)).
    All models construct their
    four VQCs and the output layer in the same order, so an equal
    `seed` gives equal initial parameters. Global RNG state is left untouched when `seed` is given.
    """
    if model not in QUANTUM_MODELS:
        raise ValueError(f"model must be one of {QUANTUM_MODELS}, got {model!r}")

    def build():
        if model == "qlstm":
            cell = CustomQLSTMCell(input_size, hidden_size, output_size, qnn_depth).float()
            return CustomLSTM(input_size, hidden_size, cell).float()
        if model in FK_CELLS:
            cell = FK_CELLS[model](input_size, hidden_size, output_size, qnn_depth).float()
            return CustomLSTM(input_size, hidden_size, cell).float()
        cell = QSLSTM_CELLS[model](input_size, hidden_size, output_size, qnn_depth, gate_epsilon=gate_epsilon).float()
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


def match_recurrence(model, recurrence):
    """Make a freshly built model follow the recurrence a saved run was trained with.

    The current qslstm / qslstm_log cells amplify the forget gate; runs tagged with a sigmoid-forget
    recurrence (2026-09-27 .. 2026-09-29) are replayed with sigmoid(q_f). Returns `model`.
    """
    import functools

    from .q_slstm_cell import QSLSTM_SIGMOID_FORGET_RECURRENCE, sigmoid_log_forget
    from .q_slstm_log_cell import QSLSTM_LOG_SIGMOID_FORGET_RECURRENCES, logarithmic_memory_update

    cell = getattr(getattr(model, "core", model), "cell", None)
    if isinstance(cell, CustomQsLSTMLogCell):
        if recurrence in QSLSTM_LOG_SIGMOID_FORGET_RECURRENCES:
            cell.memory_update = functools.partial(logarithmic_memory_update, amplified_forget=False)
    elif isinstance(cell, CustomQsLSTMCell) and recurrence == QSLSTM_SIGMOID_FORGET_RECURRENCE:
        cell.log_forget = sigmoid_log_forget
    return model


def recurrence_tag(model):
    """Recurrence version recorded in run configs (`qslstm_recurrence`); None for qlstm."""
    from .q_slstm_cell import QSLSTM_RECURRENCE
    from .q_slstm_log_cell import QSLSTM_LOG_RECURRENCE
    from .q_slstm_sqrt_cell import QSLSTM_SQRT_RECURRENCE

    tags = {"qslstm": QSLSTM_RECURRENCE, "qslstm_log": QSLSTM_LOG_RECURRENCE, "qslstm_sqrt": QSLSTM_SQRT_RECURRENCE,
            **{m: cell.recurrence for m, cell in FK_CELLS.items()}}
    return tags.get(model)
