import pytest
import torch

from q_slstm.experiments.scalar_tasks import MODELS, build_model, resolve_config
from q_slstm.models.factory import QUANTUM_MODELS, build_quantum_model
from q_slstm.models.fk_q_slstm_cell import FK_QSLSTM_RECURRENCE, CustomFkQsLSTMCell


def test_forward_shapes_and_two_state_contract():
    model = build_quantum_model("fk_qslstm", input_size=1, hidden_size=2, output_size=1, qnn_depth=1, seed=0)
    assert isinstance(model.cell, CustomFkQsLSTMCell)
    outputs, (h, c) = model(torch.randn(3, 5, 1))
    assert outputs.shape == (3, 5, 1)
    assert h.shape == c.shape == (3, 2)


def test_shares_vqc_and_output_init_with_qlstm():
    fk = build_quantum_model("fk_qslstm", 1, 2, 1, 1, seed=7).cell
    ql = build_quantum_model("qlstm", 1, 2, 1, 1, seed=7).cell
    for gate in ("input_gate", "forget_gate", "cell_gate", "output_gate"):
        assert torch.equal(getattr(fk, gate).weights, getattr(ql, gate).weights)
    assert torch.equal(fk.output_post_processing.weight, ql.output_post_processing.weight)


def test_gradients_reach_vqcs_and_classical_encoders():
    model = build_quantum_model("fk_qslstm", 1, 2, 1, 1, seed=0)
    model(torch.randn(2, 3, 1))[0].sum().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name


def test_available_in_scalar_tasks_only():
    assert "fk_qslstm" in MODELS
    assert "fk_qslstm" not in QUANTUM_MODELS  # nearest-neighbor / solar need diagnostics it lacks
    config = resolve_config({"model": "fk_qslstm", "task": "ema", "seed": 1, "scale": "pilot"})
    assert config["qslstm_recurrence"] == FK_QSLSTM_RECURRENCE
    assert config["n_qubits"] == 1 + config["hidden_size"]
    assert isinstance(build_model(config).cell, CustomFkQsLSTMCell)


def test_rejected_by_diagnostics():
    model = build_quantum_model("fk_qslstm", 1, 2, 1, 1, seed=0)
    with pytest.raises(ValueError):
        model(torch.randn(2, 3, 1), return_diagnostics=True)
