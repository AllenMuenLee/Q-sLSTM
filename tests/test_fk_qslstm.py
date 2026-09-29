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


def test_available_in_every_experiment_with_the_shared_configuration():
    from q_slstm.experiments import nearest_neighbor, solar_generation  # noqa: F401  (import check)
    from q_slstm.experiments.nearest_neighbor import resolve_config as nn_config

    assert "fk_qslstm" in QUANTUM_MODELS and "fk_qslstm" in MODELS
    config = resolve_config({"model": "fk_qslstm", "task": "ema", "seed": 1, "scale": "pilot"})
    reference = resolve_config({"model": "qslstm", "task": "ema", "seed": 1, "scale": "pilot"})
    for key in ("hidden_size", "qnn_depth", "n_qubits", "lr", "epochs", "batch_size", "sequence_length", "seeds"):
        assert config[key] == reference[key], key
    nn = nn_config({"model": "fk_qslstm", "scale": "pilot", "seed": 0})
    assert nn["qslstm_recurrence"] == FK_QSLSTM_RECURRENCE
    assert config["qslstm_recurrence"] == FK_QSLSTM_RECURRENCE
    assert config["n_qubits"] == 1 + config["hidden_size"]
    assert isinstance(build_model(config).cell, CustomFkQsLSTMCell)


def test_write_proportion_diagnostics_leave_outputs_unchanged():
    model = build_quantum_model("fk_qslstm", 1, 2, 1, 1, seed=0)
    x = torch.randn(2, 3, 1)
    with torch.no_grad():
        plain, _ = model(x)
        outputs, _, diag = model(x, return_diagnostics=True)
    torch.testing.assert_close(outputs, plain)
    assert diag["alpha"].shape == (2, 3, 2)
    assert ((diag["alpha"] > 0) & (diag["alpha"] <= 1)).all()


def test_fk_qlstm_shares_every_initial_parameter_with_fk_qslstm():
    from q_slstm.models.fk_qlstm_cell import FK_QLSTM_RECURRENCE, CustomFkQLSTMCell

    a = build_quantum_model("fk_qlstm", 1, 2, 1, 1, seed=5)
    b = build_quantum_model("fk_qslstm", 1, 2, 1, 1, seed=5)
    assert isinstance(a.cell, CustomFkQLSTMCell) and a.cell.recurrence == FK_QLSTM_RECURRENCE
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


def test_fk_models_differ_only_in_input_and_forget_activations():
    qlstm = build_quantum_model("fk_qlstm", 1, 2, 1, 1, seed=5).cell
    qslstm = build_quantum_model("fk_qslstm", 1, 2, 1, 1, seed=5).cell
    combined = torch.randn(4, 3)
    with torch.no_grad():
        i_a, f_a, g_a, o_a = qlstm.gate_values(combined)
        i_b, f_b, g_b, o_b = qslstm.gate_values(combined)
    torch.testing.assert_close(torch.logit(i_a), torch.log(i_b))
    torch.testing.assert_close(torch.logit(f_a), torch.log(f_b))
    torch.testing.assert_close(g_a, g_b)
    torch.testing.assert_close(o_a, o_b)
    assert ((i_a > 0) & (i_a < 1) & (f_a > 0) & (f_a < 1)).all()


def test_fk_qlstm_in_every_experiment():
    from q_slstm.experiments.nearest_neighbor import resolve_config as nn_config
    from q_slstm.models.fk_qlstm_cell import FK_QLSTM_RECURRENCE

    assert "fk_qlstm" in QUANTUM_MODELS
    assert resolve_config({"model": "fk_qlstm", "task": "ema", "seed": 1, "scale": "pilot"})["qslstm_recurrence"] \
        == FK_QLSTM_RECURRENCE
    assert nn_config({"model": "fk_qlstm", "scale": "pilot", "seed": 0})["qslstm_recurrence"] == FK_QLSTM_RECURRENCE
    model = build_quantum_model("fk_qlstm", 1, 2, 1, 1, seed=0)
    outputs, _, diag = model(torch.randn(2, 3, 1), return_diagnostics=True)
    assert outputs.shape == (2, 3, 1) and diag["alpha"].shape == (2, 3, 2)
