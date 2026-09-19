import importlib.util
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "experiments" / "time_series" / "train_timeseries.py"


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("train_timeseries_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_args(**overrides):
    args = dict(
        model="qslstm",
        input_size=2,
        input_projection_size=6,
        hidden_size=2,
        output_size=1,
        qnn_depth=1,
        device="cpu",
        gate_epsilon=1e-5,
    )
    args.update(overrides)
    return Namespace(**args)


def unwrap(model):
    return getattr(model, "base_model", model)


def test_qslstm_without_projection_uses_original_width(script):
    model = script.make_model(make_args(input_size=2))
    base = unwrap(model)

    assert isinstance(base, script.CustomQsLSTM)
    assert not hasattr(model, "input_projection")
    assert base.cell.n_qubits == 2 + 2
    assert base.cell.input_gate.weights.shape == (1, 4)
    assert base.cell.gate_epsilon == 1e-5


def test_qslstm_projection_determines_qubit_count(script):
    model = script.make_model(make_args(input_size=10, input_projection_size=3))
    cell = model.base_model.cell

    assert model.input_projection.in_features == 10
    assert model.input_projection.out_features == 3
    assert cell.n_qubits == 3 + 2
    for gate in (cell.input_gate, cell.forget_gate, cell.cell_gate, cell.output_gate):
        assert gate.weights.shape == (1, 5)
        assert gate.n_class == 2


def test_qslstm_projection_returns_same_structure_and_forwards_state(script):
    model = script.make_model(make_args(input_size=10, input_projection_size=3))
    x = torch.randn(2, 4, 10)

    outputs, state = model(x)
    assert outputs.shape == (2, 4, 1)
    assert len(state) == 4 and all(s.shape == (2, 2) for s in state)

    out_a, state_a = model(x[:, :2])
    out_b, _ = model(x[:, 2:], state_a)
    assert torch.allclose(torch.cat([out_a, out_b], dim=1), outputs, atol=1e-5)


def test_four_vqcs_are_independent_parameters(script):
    cell = unwrap(script.make_model(make_args())).cell
    weights = [g.weights for g in (cell.input_gate, cell.forget_gate, cell.cell_gate, cell.output_gate)]
    assert len({id(w) for w in weights}) == 4
    assert not torch.equal(weights[0], weights[1])


def test_qlstm_baseline_is_conventional_two_state_model(script):
    model = script.make_model(make_args(model="qlstm", input_size=2))
    cell = model.cell

    assert isinstance(model, script.CustomLSTM)
    assert isinstance(cell, script.CustomQLSTMCell)
    assert not hasattr(cell, "gate_epsilon")
    assert cell.n_qubits == 2 + 2

    outputs, state = model(torch.randn(3, 4, 2))
    assert outputs.shape == (3, 4, 1)
    assert len(state) == 2 and all(s.shape == (3, 2) for s in state)


def test_qlstm_projection_determines_qubit_count(script):
    model = script.make_model(make_args(model="qlstm", input_size=10, input_projection_size=3))
    cell = model.base_model.cell

    assert model.input_projection.out_features == 3
    assert cell.n_qubits == 3 + 2
    assert cell.input_gate.weights.shape == (1, 5)

    outputs, state = model(torch.randn(2, 4, 10))
    assert outputs.shape == (2, 4, 1) and len(state) == 2


def test_qlstm_and_qslstm_are_parameter_matched_and_identically_initialized(script):
    def build(name):
        torch.manual_seed(123)
        return script.make_model(make_args(model=name, input_size=10, input_projection_size=3))

    qlstm, qslstm = build("qlstm"), build("qslstm")

    count = lambda m: sum(p.numel() for p in m.parameters())
    assert count(qlstm) == count(qslstm)

    shared = qlstm.state_dict()
    other = qslstm.state_dict()
    assert shared.keys() == other.keys()
    assert all(torch.equal(shared[k], other[k]) for k in shared)

    # equal initial values, but never tied: training one must not move the other
    p1 = next(qlstm.parameters())
    p2 = next(qslstm.parameters())
    assert p1.data_ptr() != p2.data_ptr()


@pytest.mark.parametrize("input_size, projection", [(3, 6), (12, 4)])
def test_classical_lstm_keeps_two_state_contract(script, input_size, projection):
    model = script.make_model(make_args(model="lstm", input_size=input_size, input_projection_size=projection))
    assert isinstance(unwrap(model), script.CustomLSTM)

    x = torch.randn(3, 5, input_size)
    outputs, state = model(x)

    assert outputs.shape == (3, 5, 1)
    assert len(state) == 2 and all(s.shape == (3, 2) for s in state)
    outputs.sum().backward()


def test_all_models_smoke_forward_backward(script):
    for name in ("qlstm", "qslstm", "lstm"):
        model = script.make_model(make_args(model=name, input_size=2))
        outputs, _ = model(torch.randn(3, 4, 2))
        prediction = outputs[:, -1, :]
        assert prediction.shape == (3, 1)
        loss = prediction.pow(2).mean()
        loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_cli_exposes_gate_epsilon():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0
    assert "--gate_epsilon" in result.stdout
