"""Configuration, training protocol, resume / identity checks, failures, and tiny end-to-end pilots."""

import importlib.util
import json

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from solar_test_utils import EXPERIMENT_DIR
from q_slstm.experiments import solar_generation as ex
from q_slstm.datasets.solar_generation import PreparedDataset

TINY = dict(scale="pilot", hidden_size=1, qnn_depth=1, batch_size=32, epochs=2)


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"solar_{name}_under_test", EXPERIMENT_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tiny_config(pointer, tmp_path, model="qlstm", seed=0, **overrides):
    return ex.resolve_config({**TINY, **overrides, "model": model, "seed": seed,
                              "dataset_manifest": str(pointer), "save_dir": str(tmp_path)})


def test_presets_match_the_documented_study_settings():
    assert ex.PRESETS["paper"] == dict(sequence_length=32, projection_size=3, hidden_size=6, qnn_depth=3,
                                       batch_size=64, epochs=100, lr=1e-2, n_seeds=5)
    assert ex.PRESETS["pilot"] == dict(sequence_length=8, projection_size=3, hidden_size=2, qnn_depth=1,
                                       batch_size=8, epochs=2, lr=1e-2, n_seeds=2)


def test_log_variant_records_its_recurrence(synthetic_pointer, tmp_path):
    from q_slstm.models.q_slstm_log_cell import QSLSTM_LOG_RECURRENCE

    config = tiny_config(synthetic_pointer, tmp_path, "qslstm_log")
    assert config["study"]["qslstm_recurrence"] == QSLSTM_LOG_RECURRENCE
    assert ex.run_directory(config).name == "qslstm_log"


def test_config_resolution_labels_overrides_and_rejects_conflicts(synthetic_pointer, tmp_path):
    a = tiny_config(synthetic_pointer, tmp_path, "qlstm", 0)
    b = tiny_config(synthetic_pointer, tmp_path, "qslstm", 3)
    assert a["study_id"] == b["study_id"] and a["study"] == b["study"]  # model / seed excluded from study id
    assert a["config_hash"] != b["config_hash"]
    assert a["study"]["scale_label"] == "pilot-overridden"  # synthetic dates, L=4, tiny model
    assert a["study"]["n_qubits"] == 3 + 1 and a["study"]["raw_input_size"] == 13
    assert a["study"]["quantum_input_size"] == 3 and a["study"]["projection_activation"] == "tanh"
    c = tiny_config(synthetic_pointer, tmp_path, lr=0.02)
    assert c["study_id"] != a["study_id"] and c["study"]["training_overrides"]["lr"]["used"] == 0.02
    assert ex.run_directory(a).parts[-4:] == ("pilot-overridden", a["study_id"], "seed_0", "qlstm")
    with pytest.raises(ValueError, match="horizon"):
        tiny_config(synthetic_pointer, tmp_path, horizon=2)
    with pytest.raises(ValueError, match="raw input width"):
        tiny_config(synthetic_pointer, tmp_path, raw_input_size=3)
    with pytest.raises(ValueError, match="prepared for scale"):
        tiny_config(synthetic_pointer, tmp_path, scale="paper")
    with pytest.raises(ValueError, match="model"):
        tiny_config(synthetic_pointer, tmp_path, model="lstm")
    assert a["seeds"]["data_seed"] is None and "inapplicable" in a["seeds"]["data_seed_note"]


def test_epoch_orders_are_deterministic_and_independent_of_model(synthetic_pointer, tmp_path):
    from q_slstm.utils.seeds import epoch_permutation

    a, b = tiny_config(synthetic_pointer, tmp_path, "qlstm"), tiny_config(synthetic_pointer, tmp_path, "qslstm")
    assert a["seeds"]["loader"] == b["seeds"]["loader"]
    p1 = epoch_permutation(a["seeds"]["loader"], 1, 50)
    assert (p1 == epoch_permutation(b["seeds"]["loader"], 1, 50)).all()
    assert not (p1 == epoch_permutation(a["seeds"]["loader"], 2, 50)).all()


# ---------------------------------------------------------------------------------------------
# 12. Training protocol, checkpoint selection, identity, failures
# ---------------------------------------------------------------------------------------------

class Stub(nn.Module):
    """Tiny torch model with the wrapper interface; counts every window it sees."""

    def __init__(self, nan=False):
        super().__init__()
        self.lin = nn.Linear(13, 1)
        self.projection, self.n_qubits = self.lin, 0  # attributes the run artifacts record
        self.nan, self.seen = nan, 0

    def forward(self, x, hidden=None, return_diagnostics=False):
        self.seen += x.shape[0]
        out = self.lin(x) * (float("nan") if self.nan else 1.0)
        if return_diagnostics:
            return out, None, {"alpha": torch.full((*x.shape[:2], 2), 0.5)}
        return out, None


def _train_setup(pointer, tmp_path, **overrides):
    config = tiny_config(pointer, tmp_path, **overrides)
    data = PreparedDataset(pointer)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "scaler.json").write_text(json.dumps(data.scaler))
    return config, data, run_dir


def test_training_runs_full_budget_selects_on_validation_only_earliest_tie(synthetic_pointer, tmp_path, monkeypatch):
    config, data, run_dir = _train_setup(synthetic_pointer, tmp_path, epochs=4, batch_size=36)
    scripted = iter([3.0, 1.0, 1.0, 2.0])
    monkeypatch.setattr(ex, "evaluate_mse", lambda model, dataset, config, epoch: next(scripted))
    model = Stub()
    train, val = data.torch_dataset("train"), data.torch_dataset("val")
    result = ex.train_model(model, train, val, config, run_dir)
    assert result["epochs_run"] == 4 and result["best_epoch"] == 2  # earliest of the tied minimum
    assert model.seen == 4 * len(train)  # the test split is never touched during training
    history = pd.read_csv(run_dir / "history.csv")
    assert history["is_best"].tolist() == [True, True, False, False]
    assert torch.load(run_dir / "checkpoints" / "best.pt", weights_only=False)["epoch"] == 2
    last = torch.load(run_dir / "checkpoints" / "last.pt", weights_only=False)
    assert last["epoch"] == 4 and "optimizer_state_dict" in last and "torch_rng_state" in last
    assert history["n_train_windows"].eq(len(train)).all()  # sample-weighted incl. partial final batch
    assert len(train) % config["study"]["batch_size"] != 0


def test_partial_batches_are_sample_weighted(synthetic_pointer, tmp_path):
    config, data, _ = _train_setup(synthetic_pointer, tmp_path, batch_size=7)
    model = Stub()
    val = data.torch_dataset("val")
    got = ex.evaluate_mse(model, val, config, 1)
    inputs, target = ex.batch_tensors(val, np.arange(len(val)))
    with torch.no_grad():
        expected = float(((model(inputs)[0][:, -1, :] - target) ** 2).mean())
    assert got == pytest.approx(expected, rel=1e-6)


def test_non_finite_outputs_abort_with_run_epoch_batch_identity(synthetic_pointer, tmp_path):
    config, data, run_dir = _train_setup(synthetic_pointer, tmp_path, seed=5)
    with pytest.raises(ex.NonFiniteError, match=r"run_seed=5 model=qlstm epoch=1 batch=0"):
        ex.train_model(Stub(nan=True), data.torch_dataset("train"), data.torch_dataset("val"), config, run_dir)


def test_failed_runs_leave_failure_json_and_no_completion(synthetic_pointer, tmp_path, monkeypatch):
    config = tiny_config(synthetic_pointer, tmp_path, seed=9)
    monkeypatch.setattr(ex, "build_projected_quantum_model", lambda *a, **k: Stub(nan=True))
    with pytest.raises(ex.NonFiniteError):
        ex.run_experiment(config)
    run_dir = ex.run_directory(config)
    assert (run_dir / "failure.json").exists() and not (run_dir / "complete.json").exists()
    assert ex.run_status(config) == "incomplete"
    failure = json.loads((run_dir / "failure.json").read_text())
    assert failure["error_type"] == "NonFiniteError" and failure["run_seed"] == 9


def test_resume_and_overwrite_require_matching_identity(synthetic_pointer, tmp_path):
    config = tiny_config(synthetic_pointer, tmp_path)
    run_dir = ex.run_directory(config)
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps({**config, "config_hash": "different"}))
    assert ex.run_status(config) == "incompatible"
    with pytest.raises(ex.IncompatibleRunError, match="different configuration"):
        ex.run_experiment(config, resume=True)
    (run_dir / "config.json").write_text(json.dumps(config))
    (run_dir / "complete.json").write_text(json.dumps({"config_hash": config["config_hash"],
                                                       "dataset_id": config["study"]["dataset_id"],
                                                       "artifact_sha256": {"history.csv": "0" * 64}}))
    assert ex.run_status(config) == "incomplete"  # complete.json alone is not enough


# ---------------------------------------------------------------------------------------------
# 13. Real VQC runs, resume, and end-to-end analysis
# ---------------------------------------------------------------------------------------------

@pytest.mark.slow
def test_interrupted_run_resumes_to_the_same_result(synthetic_pointer, tmp_path, monkeypatch):
    full = tiny_config(synthetic_pointer, tmp_path / "full", "qslstm")
    ex.run_experiment(full)
    part = tiny_config(synthetic_pointer, tmp_path / "part", "qslstm")
    real_eval, calls = ex.evaluate_mse, {"n": 0}

    def interrupt_second_validation(*args):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt("simulated interruption during epoch 2")
        return real_eval(*args)

    monkeypatch.setattr(ex, "evaluate_mse", interrupt_second_validation)
    with pytest.raises(KeyboardInterrupt):
        ex.run_experiment(part)
    monkeypatch.setattr(ex, "evaluate_mse", real_eval)
    assert ex.run_status(part) == "incomplete"
    ex.run_experiment(part, resume=True)
    assert ex.run_status(part) == "complete"
    summary = json.loads((ex.run_directory(part) / "train_summary.json").read_text())
    assert summary["resumed_from_epoch"] == 1
    a = pd.read_csv(ex.run_directory(full) / "predictions_test.csv")
    b = pd.read_csv(ex.run_directory(part) / "predictions_test.csv")
    np.testing.assert_allclose(a["prediction_mwh"], b["prediction_mwh"], rtol=1e-6)
    with pytest.raises(ex.IncompatibleRunError, match="already complete"):
        ex.run_experiment(part)
    assert ex.run_experiment(part, resume=True) == ex.run_directory(part)  # verified skip


@pytest.fixture(scope="module")
def sweep_root(synthetic_pointer, tmp_path_factory):
    root = tmp_path_factory.mktemp("solar_sweep")
    sweep = load_script("run_sweep")
    common = ["--scale", "pilot", "--dataset-manifest", str(synthetic_pointer), "--hidden-size", "1",
              "--qnn-depth", "1", "--batch-size", "32", "--epochs", "1", "--seeds", "0", "1",
              "--workers", "2", "--save-dir", str(root), "--save-alpha-trace"]
    assert sweep.main(["--model", "qlstm"] + common) == 0
    study = next((root / "pilot-overridden").iterdir())
    analyze = load_script("analyze_results")
    with pytest.raises(SystemExit, match="qslstm"):
        analyze.analyze(study, root / "early")  # only one model finished so far
    assert sweep.main(["--model", "qslstm"] + common) == 0
    with pytest.raises(SystemExit, match="already complete"):
        sweep.main(["--model", "qslstm"] + common)  # refuses accidental overwrite
    assert sweep.main(["--model", "qslstm", "--resume"] + common) == 0  # verified runs are skipped
    return study, analyze.analyze(study)


REQUIRED = ex.REQUIRED_ARTIFACTS + ("complete.json", "console_log.txt", "alpha_trace_test.csv")


@pytest.mark.slow
def test_pilot_runs_write_consistent_artifacts(sweep_root, synthetic_pointer):
    study, _ = sweep_root
    data = PreparedDataset(synthetic_pointer)
    n_test = len(data.split_windows("test"))
    assert json.loads((study / "study_manifest.json").read_text())["study"]["dataset_id"] == data.dataset_id
    assert json.loads((study / "seeds_qslstm.json").read_text())["seeds"] == [0, 1]
    for seed in (0, 1):
        for model in ("qlstm", "qslstm"):
            run_dir = study / f"seed_{seed}" / model
            for name in REQUIRED:
                assert (run_dir / name).exists(), (run_dir, name)
            assert not (run_dir / "failure.json").exists()
            preds = pd.read_csv(run_dir / "predictions_test.csv")
            assert list(preds.columns[:len(ex.PREDICTION_COLUMNS)]) == list(ex.PREDICTION_COLUMNS)
            assert len(preds) == n_test and preds["window_id"].is_unique
            np.testing.assert_allclose(preds["squared_error"], (preds["prediction_mwh"] - preds["target_mwh"]) ** 2)
            assert preds["alpha_final_mean"].between(0, 1 + 1e-5).all()
            params = json.loads((run_dir / "parameters.json").read_text())
            assert (params["raw_input_size"], params["quantum_input_size"], params["n_qubits"]) == (13, 3, 4)
            for csv in run_dir.glob("*.csv"):
                cols = " ".join(pd.read_csv(csv, nrows=0).columns).lower()
                assert "input_gate" not in cols and "forget_gate" not in cols and "raw_gate" not in cols


@pytest.mark.slow
def test_analysis_combines_paired_runs(sweep_root):
    study, out = sweep_root
    for name in ("combined_seed_metrics.csv", "paired_comparison.csv", "pairing_checks.json", "summary.md",
                 "primary_paired_mse.csv", "model_summary.csv", "runtime.csv"):
        assert (out / name).exists(), name
    plots = {p.name for p in (out / "plots").glob("*.png")}
    assert {"seed_mse.png", "paired_mse_differences.png", "learning_curves.png", "monthly_daylight_error.png",
            "trace_test.png", "alpha_by_input_step.png"} <= plots
    checks = json.loads((out / "pairing_checks.json").read_text())
    assert set(checks) == {"0", "1"} and all(c["accepted"] for c in checks.values())
    primary = pd.read_csv(out / "primary_paired_mse.csv")
    assert primary["run_seed"].tolist() == [0, 1]
    np.testing.assert_allclose(primary["difference"], primary["qslstm"] - primary["qlstm"])
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "pilot study" in summary and "completed and accepted pairs: 2" in summary


@pytest.mark.slow
def test_analysis_refuses_to_pool_studies(sweep_root, tmp_path):
    import shutil

    study, _ = sweep_root
    shutil.copytree(study, tmp_path / "a")
    cfg_path = tmp_path / "a" / "seed_0" / "qlstm" / "config.json"
    cfg = json.loads(cfg_path.read_text())
    shutil.copytree(study / "seed_0", tmp_path / "b" / "seed_0")
    cfg["study_id"] = "other"
    (tmp_path / "b" / "seed_0" / "qlstm" / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(SystemExit, match="several studies"):
        load_script("analyze_results").analyze(tmp_path)


def test_training_and_sweep_cli_expose_required_arguments():
    help_text = load_script("train_solar_generation").build_parser().format_help()
    for flag in ("--dataset-manifest", "--model", "--scale", "--seed", "--hidden-size", "--projection-size",
                 "--qnn-depth", "--gate-epsilon", "--batch-size", "--epochs", "--lr", "--weight-decay",
                 "--grad-clip", "--device", "--save-dir", "--horizon", "--raw-input-size"):
        assert flag in help_text, flag
    sweep = load_script("run_sweep")
    args = sweep.build_parser().parse_args(["--model", "qslstm", "--jobs", "3", "--resume", "--seeds", "4", "2"])
    assert args.workers == 3
    argv = sweep.child_arguments(args, 4)
    assert argv[:4] == ["--model", "qslstm", "--seed", "4"] and "--resume" in argv
    assert not {"--workers", "--seeds", "--n-seeds", "--master-seed", "--dry-run"} & set(argv)
    for name in ("collect_data", "prepare_dataset"):
        text = load_script(name).build_parser().format_help()
        for flag in ("--scale", "--data-dir", "--offline"):
            assert flag in text, (name, flag)
    collect_help = load_script("collect_data").build_parser().format_help()
    assert all(f in collect_help for f in ("--start-date", "--end-date", "--locations-file", "--refresh"))
    prep_help = load_script("prepare_dataset").build_parser().format_help()
    assert "--sequence-length" in prep_help and "--min-coverage" in prep_help
