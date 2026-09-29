"""Q-sLSTM cells amplify the forget gate; runs from the sigmoid-forget period replay as they were trained."""

import pytest
import torch

from q_slstm.models.factory import build_quantum_model, match_recurrence, recurrence_tag
from q_slstm.models.q_slstm_cell import (
    QSLSTM_RECURRENCE,
    QSLSTM_SIGMOID_FORGET_RECURRENCE,
    bounded_log_ratio,
    sigmoid_log_forget,
)
from q_slstm.models.q_slstm_log_cell import (
    QSLSTM_LOG_FLOAT32_GATES_RECURRENCE,
    QSLSTM_LOG_RECURRENCE,
    QSLSTM_LOG_SIGMOID_FORGET_F64_RECURRENCE,
)


def test_tags_name_the_amplified_recurrences():
    assert recurrence_tag("qslstm") == QSLSTM_RECURRENCE == "xlstm_stabilized_v1"
    assert recurrence_tag("qslstm_log") == QSLSTM_LOG_RECURRENCE == "log_gate_binary_scale_f64_gates_v2"
    assert recurrence_tag("qlstm") is None


def test_qslstm_forget_gate_is_amplified():
    cell = build_quantum_model("qslstm", 1, 2, 1, 1, seed=0).cell
    q = torch.tensor([[-.5, .8]])
    torch.testing.assert_close(cell.log_forget(q), bounded_log_ratio(q, cell.gate_epsilon))


@pytest.mark.parametrize("model", ["qslstm", "qslstm_log"])
def test_current_runs_are_left_as_built(model):
    net = build_quantum_model(model, 1, 2, 1, 1, seed=0)
    match_recurrence(net, recurrence_tag(model))
    assert "log_forget" not in vars(net.cell) and "memory_update" not in vars(net.cell)


def test_sigmoid_forget_qslstm_runs_replay_with_sigmoid():
    net = match_recurrence(build_quantum_model("qslstm", 1, 2, 1, 1, seed=0), QSLSTM_SIGMOID_FORGET_RECURRENCE)
    q = torch.tensor([[-.5, .8]])
    torch.testing.assert_close(net.cell.log_forget(q), sigmoid_log_forget(q))


@pytest.mark.parametrize("tag", [QSLSTM_LOG_SIGMOID_FORGET_F64_RECURRENCE, QSLSTM_LOG_FLOAT32_GATES_RECURRENCE])
def test_sigmoid_forget_log_runs_replay_with_sigmoid(tag):
    x = torch.randn(2, 4, 1)
    legacy = match_recurrence(build_quantum_model("qslstm_log", 1, 2, 1, 1, seed=3), tag)
    current = build_quantum_model("qslstm_log", 1, 2, 1, 1, seed=3)
    with torch.no_grad():
        assert not torch.allclose(legacy(x)[0], current(x)[0])
    assert legacy.cell.memory_update.keywords == {"amplified_forget": False}
