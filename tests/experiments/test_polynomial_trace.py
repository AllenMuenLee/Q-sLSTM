"""Historical trace tooling must follow the recurrence recorded by a new run."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from q_slstm.experiments.nearest_neighbor import make_datasets, resolve_config, run_experiment
from q_slstm.models.sequence_wrappers import CustomQsLSTM
from qslstm_test_utils import make_cell


def test_gate_trace_matches_production_stabilized_rollout(monkeypatch):
    scripts = Path(__file__).resolve().parents[2] / "scripts/experiments/nearest_neighbor"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("polynomial_gate_trace_test", scripts / "gate_trace.py")
    trace = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trace)
    config = resolve_config({"model": "qslstm", "scale": "pilot", "seed": 0})
    with pytest.raises(ValueError, match="fresh v2 run"):
        trace.trace_run(Path("unused"), {**config, "qslstm_recurrence": "polynomial_binary_scale_v1"}, "test", 2)
    model = CustomQsLSTM(3, 2, make_cell(3, 2))
    monkeypatch.setattr(trace, "load_model", lambda *_: model)
    dataset = make_datasets(config)["test"]
    with torch.no_grad():
        expected, _, diagnostics = model(dataset.tensors["inputs"][:2], return_diagnostics=True)
    mask = dataset.tensors["metric_mask"][:2, :, 0].numpy()
    frame = trace.trace_run(Path("unused"), config, "test", 2)
    expected_predictions = np.repeat(expected[:, :, 0].numpy()[mask], 2)
    expected_alpha = diagnostics["alpha"].numpy()[mask].reshape(-1)
    np.testing.assert_allclose(frame["pred"], expected_predictions, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(frame["alpha"], expected_alpha, rtol=1e-6, atol=1e-7)
    assert set(frame["scale_kind"]) == {"log_stabilizer"}


def test_gate_trace_matches_production_log_gate_rollout(monkeypatch):
    from q_slstm.models.q_slstm_log_cell import CustomQsLSTMLogCell

    scripts = Path(__file__).resolve().parents[2] / "scripts/experiments/nearest_neighbor"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("log_gate_trace_test", scripts / "gate_trace.py")
    trace = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trace)
    config = resolve_config({"model": "qslstm_log", "scale": "pilot", "seed": 0})
    stub = make_cell(3, 2)
    cell = CustomQsLSTMLogCell(3, 2, 1, vqc_depth=1)
    for name in ("input_gate", "forget_gate", "cell_gate", "output_gate", "output_post_processing"):
        setattr(cell, name, getattr(stub, name))
    model = CustomQsLSTM(3, 2, cell)
    monkeypatch.setattr(trace, "load_model", lambda *_: model)
    dataset = make_datasets(config)["test"]
    with torch.no_grad():
        expected, _, diagnostics = model(dataset.tensors["inputs"][:2], return_diagnostics=True)
    mask = dataset.tensors["metric_mask"][:2, :, 0].numpy()
    frame = trace.trace_run(Path("unused"), config, "test", 2)
    np.testing.assert_allclose(frame["pred"], np.repeat(expected[:, :, 0].numpy()[mask], 2), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(frame["alpha"], diagnostics["alpha"].numpy()[mask].reshape(-1), rtol=1e-6, atol=1e-7)
    assert set(frame["scale_kind"]) == {"binary_exponent"}


@pytest.mark.parametrize("previous_version", [None, "polynomial_binary_scale_v1"])
def test_new_run_cannot_overwrite_legacy_results(tmp_path, previous_version):
    config = resolve_config({"model": "qslstm", "scale": "pilot", "seed": 0})
    previous = {k: v for k, v in config.items() if k != "qslstm_recurrence"}
    if previous_version is not None:
        previous["qslstm_recurrence"] = previous_version
    (tmp_path / "config.json").write_text(json.dumps(previous), encoding="utf-8")
    (tmp_path / "complete.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="different Q-sLSTM recurrence"):
        run_experiment(config, tmp_path)
    assert (tmp_path / "complete.json").read_text(encoding="utf-8") == "{}"
    assert json.loads((tmp_path / "config.json").read_text(encoding="utf-8")) == previous
