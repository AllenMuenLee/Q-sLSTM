"""End-to-end pipeline: config resolution, pairing, tiny pilot runs, artifacts, analysis, extrapolation."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from q_slstm.experiments import nearest_neighbor as nn_exp
from q_slstm.experiments.nearest_neighbor import (
    NonFiniteError,
    make_datasets,
    predict_dataset,
    resolve_config,
    run_experiment,
    train_model,
    verify_pairing,
)
from q_slstm.models.factory import build_quantum_model

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts" / "experiments" / "nearest_neighbor"

TINY = dict(scale="pilot", sequence_length=6, train_size=12, val_fraction=0.25, test_size=4,
            extrapolation_length=10, extrapolation_size=4, hidden_size=2, qnn_depth=1,
            batch_size=6, epochs=2, run_extrapolation=True)


def load_script(name):
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(f"nn_{name}_under_test", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tiny_config(model, tmp_path, seed=0, **overrides):
    return resolve_config({**TINY, **overrides, "model": model, "seed": seed, "save_dir": str(tmp_path)})


def test_paper_preset_matches_the_documented_scale(tmp_path):
    cfg = resolve_config({"model": "qslstm", "seed": 0, "save_dir": str(tmp_path)})
    assert cfg["scale_label"] == "paper" and cfg["preset_overrides"] == {}
    assert (cfg["sequence_length"], cfg["n_candidates"]) == (32, 31)
    assert (cfg["train_size"], cfg["optimizer_train_size"], cfg["val_size"], cfg["test_size"]) == (4000, 3500, 500, 1000)
    assert (cfg["extrapolation_length"], cfg["extrapolation_size"]) == (64, 250)
    assert cfg["input_size"] == 3 and cfg["output_size"] == 1 and cfg["input_projection"] is False
    assert cfg["n_qubits"] == 3 + cfg["hidden_size"]
    assert {k: round(v, 6) for k, v in cfg["split_fractions"].items()} == {"train": 0.7, "validation": 0.1, "test": 0.2}
    assert "patience" not in cfg


def test_patience_argument_is_gone():
    train = load_script("train_nearest_neighbor")
    assert "--patience" not in train.build_parser().format_help()
    with pytest.raises(SystemExit):
        train.build_parser().parse_args(["--model", "qlstm", "--patience", "10"])


def test_overrides_are_labeled_and_conflicting_sizes_rejected(tmp_path):
    cfg = resolve_config({"model": "qlstm", "scale": "paper", "train_size": 100, "save_dir": str(tmp_path)})
    assert cfg["scale_label"] == "paper-overridden" and cfg["preset_overrides"]["train_size"]["used"] == 100
    assert resolve_config({"model": "qlstm", "scale": "paper", "train_size": 4000})["scale_label"] == "paper"
    with pytest.raises(ValueError, match="input_size"):
        resolve_config({"model": "qlstm", "input_size": 5})
    with pytest.raises(ValueError, match="output_size"):
        resolve_config({"model": "qlstm", "output_size": 2})
    with pytest.raises(ValueError, match="model"):
        resolve_config({"model": "lstm"})


def test_seed_map_is_shared_by_paired_models_and_changes_across_seeds(tmp_path):
    a, b = tiny_config("qlstm", tmp_path, 1), tiny_config("qslstm", tmp_path, 1)
    c = tiny_config("qslstm", tmp_path, 2)
    assert a["seeds"] == b["seeds"] and a["seeds"] != c["seeds"]
    assert len({a["seeds"][k] for k in ("data", "split", "loader", "model")}) == 4
    d = tiny_config("qlstm", tmp_path, 1, data_seed=99)
    assert d["seeds"]["data"] != a["seeds"]["data"] and d["seeds"]["model"] == a["seeds"]["model"]


def test_paired_models_get_identical_datasets(tmp_path):
    a = make_datasets(tiny_config("qlstm", tmp_path, 3))
    b = make_datasets(tiny_config("qslstm", tmp_path, 3))
    other = make_datasets(tiny_config("qlstm", tmp_path, 4))
    for split in a:
        assert a[split].checksums() == b[split].checksums()
        assert a[split].checksums()["combined"] != other[split].checksums()["combined"]
    assert a["extrapolation"].sequence_length == 10 and a["test"].sequence_length == 6
    assert set(a["train"].sequence_ids.tolist()).isdisjoint(a["val"].sequence_ids.tolist())


def test_non_finite_values_abort_with_run_epoch_batch_identity(tmp_path):
    class NaNModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.p = nn.Parameter(torch.zeros(1))

        def forward(self, x, **kwargs):
            return (self.p * float("nan") + x[..., :1], None)

    cfg = tiny_config("qlstm", tmp_path, 5)
    ds = make_datasets(cfg)
    (tmp_path / "run").mkdir()
    with pytest.raises(NonFiniteError, match=r"run_seed=5 model=qlstm epoch=1 batch=0"):
        train_model(NaNModel(), ds["train"], ds["val"], cfg, tmp_path / "run")


def test_training_uses_full_epoch_budget_without_early_stopping(tmp_path):
    class Flat(nn.Module):  # never improves: validation MSE is constant
        def __init__(self):
            super().__init__()
            self.p = nn.Parameter(torch.zeros(1))

        def forward(self, x, **kwargs):
            return (self.p * 0.0 + x[..., :1] * 0.0, None)

    cfg = tiny_config("qlstm", tmp_path, 6, epochs=7)
    ds = make_datasets(cfg)
    (tmp_path / "run").mkdir()
    result = train_model(Flat(), ds["train"], ds["val"], cfg, tmp_path / "run")
    assert result["epochs_run"] == 7 and result["best_epoch"] == 1


@pytest.fixture(scope="module")
def pilot(tmp_path_factory):
    root = tmp_path_factory.mktemp("nn_pilot")
    dirs = {}
    for model in ("qlstm", "qslstm"):
        cfg = tiny_config(model, root, seed=0)
        dirs[model] = (cfg, run_experiment(cfg))
    return root, dirs


REQUIRED_ARTIFACTS = [
    "config.json", "dataset_manifest.json", "init_parameters.json", "parameters.json", "history.csv",
    "train_summary.json", "timing.json", "environment.json", "source_revision.json", "complete.json",
    "checkpoints/best.pt", "checkpoints/last.pt",
    "predictions_test.csv", "per_sequence_metrics_test.csv", "per_seed_metrics_test.csv",
    "predictions_extrapolation.csv", "per_seed_metrics_extrapolation.csv",
]


@pytest.mark.slow
def test_pilot_run_writes_all_artifacts_and_evaluates_all_case_types(pilot):
    _, dirs = pilot
    for model, (cfg, run_dir) in dirs.items():
        for name in REQUIRED_ARTIFACTS:
            assert (run_dir / name).exists(), f"{model}: missing {name}"
        assert not (run_dir / "failure.json").exists()
        per_seed = pd.read_csv(run_dir / "per_seed_metrics_test.csv")
        assert set(per_seed["case_type"]) == {"all", "iid", "late", "early", "near_best"}
        assert per_seed.loc[per_seed["case_type"] == "all", "n_sequences"].item() == 4
        history = pd.read_csv(run_dir / "history.csv")
        assert {"train_loss", "val_mse", "grad_norm_mean", "clip_fraction"} <= set(history.columns)
        assert history["is_best"].any()
        assert json.loads((run_dir / "parameters.json").read_text())["n_qubits"] == 3 + 2
        manifest = json.loads((run_dir / "dataset_manifest.json").read_text())
        assert manifest["splits"]["test"]["case_counts"] == {"iid": 1, "late": 1, "early": 1, "near_best": 1}
        assert manifest["splits"]["test"]["checksums"]["combined"]


@pytest.mark.slow
def test_predictions_have_required_columns_and_no_raw_gates(pilot):
    _, dirs = pilot
    required = ("run_seed model split case_type sequence_id sequence_length timestep candidate_index similarity "
                "running_best_similarity is_event is_metric_step best_index target prediction absolute_error "
                "squared_error alpha_mean").split()
    for _, (cfg, run_dir) in dirs.items():
        for name in ("predictions_test.csv", "predictions_extrapolation.csv"):
            frame = pd.read_csv(run_dir / name)
            assert list(frame.columns) == required
            L = frame["sequence_length"].iloc[0]
            assert len(frame) % L == 0
            m = frame["is_metric_step"]
            assert not m[frame["timestep"] < 2].any() and m[frame["timestep"] >= 2].all()
            assert frame.loc[frame["timestep"] == 0, "target"].isna().all()
            assert frame["alpha_mean"].between(0, 1 + 1e-5).all()
        for csv in run_dir.glob("*.csv"):
            for col in pd.read_csv(csv, nrows=0).columns:
                low = col.lower()
                assert not any(tag in low for tag in ("input_gate", "forget_gate", "i_t", "f_t", "raw_gate")), (csv, col)
        assert "gate" not in " ".join(pd.read_csv(run_dir / "per_sequence_metrics_test.csv", nrows=0).columns)


@pytest.mark.slow
def test_paired_runs_share_data_loader_order_and_initial_parameters(pilot):
    _, dirs = pilot
    check = verify_pairing(dirs["qlstm"][1], dirs["qslstm"][1])
    assert check["dataset_checksums_equal"] and check["seed_map_equal"] and check["loader_order_equal"]
    assert check["initial_shared_parameters_equal"] and check["all_parameter_names_shared"]
    counts = [json.loads((d / "parameters.json").read_text())["trainable_parameters"] for _, d in dirs.values()]
    assert counts[0] == counts[1]


@pytest.mark.slow
def test_length_trained_checkpoint_evaluates_at_longer_length_without_retraining(pilot):
    _, dirs = pilot
    for model, (cfg, run_dir) in dirs.items():
        assert cfg["sequence_length"] == 6
        history_before = (run_dir / "history.csv").read_text()
        net = build_quantum_model(model, 3, cfg["hidden_size"], 1, cfg["qnn_depth"],
                                  gate_epsilon=cfg["gate_epsilon"], seed=12345)
        net.load_state_dict(torch.load(run_dir / "checkpoints" / "best.pt")["model_state_dict"])
        long_ds = make_datasets(cfg)["extrapolation"]
        assert long_ds.sequence_length == 10
        pred, alpha = predict_dataset(net, long_ds, cfg)
        assert pred.shape == (4, 10) and alpha.shape == (4, 10)
        saved = pd.read_csv(run_dir / "predictions_extrapolation.csv")
        np.testing.assert_allclose(saved["prediction"].to_numpy(), pred.ravel(), atol=1e-6)
        assert (run_dir / "history.csv").read_text() == history_before


@pytest.mark.slow
def test_analysis_combines_paired_runs(pilot, tmp_path):
    root, dirs = pilot
    analyze = load_script("analyze_results")
    out = analyze.analyze(root, tmp_path / "analysis")
    for name in ("summary.md", "combined_seed_table.csv", "pairing_check.json", "parameter_counts.csv",
                 "paired_comparison_test.csv", "primary_paired_differences_test.csv",
                 "paired_comparison_extrapolation.csv"):
        assert (out / name).exists(), name
    plots = {p.name for p in (out / "plots").glob("*.png")}
    assert {"mse_by_case_test.png", "event_error_curves_test.png", "event_alpha_curves_test.png",
            "alpha_event_vs_distractor_test.png", "timestep_error_drift_test.png", "traces_test.png",
            "mse_by_case_extrapolation.png"} <= plots
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "scale `pilot-overridden`" in summary and "EXTRAPOLATION" in summary
    combined = pd.read_csv(out / "combined_seed_table.csv")
    assert not any("gate" in c for c in combined.columns)
    assert list(pd.read_csv(out / "primary_paired_differences_test.csv")["run_seed"]) == [0]


def _planned(sweep, capsys, argv):
    assert sweep.main(argv + ["--dry-run"]) == 0
    out = capsys.readouterr().out
    seeds = json.loads(next(l for l in out.splitlines() if l.startswith("seeds:")).split(":", 1)[1])
    dirs = [Path(l.strip()).parts[-2:] for l in out.splitlines() if l.startswith("   ")]
    return seeds, dirs, out


def test_sweep_runs_one_model_with_a_chosen_number_of_reproducible_seeds(tmp_path, capsys):
    sweep = load_script("run_sweep")
    base = ["--scale", "pilot", "--save-dir", str(tmp_path), "--n-seeds", "6", "--master-seed", "5", "--workers", "3"]
    seeds_a, dirs_a, out = _planned(sweep, capsys, ["--model", "qlstm"] + base)
    seeds_b, dirs_b, _ = _planned(sweep, capsys, ["--model", "qslstm"] + base)
    assert len(seeds_a) == 6 and len(set(seeds_a)) == 6
    assert seeds_a == seeds_b  # separate invocations of the two models reuse the same seeds
    assert dirs_a == [(f"seed_{s}", "qlstm") for s in seeds_a]
    assert dirs_b == [(f"seed_{s}", "qslstm") for s in seeds_a]
    assert "3 worker(s)" in out
    # a larger request extends the list; a different master seed changes it
    seeds_more, _, _ = _planned(sweep, capsys, ["--model", "qlstm"] + base[:4] + ["--n-seeds", "9", "--master-seed", "5"])
    assert seeds_more[:6] == seeds_a
    seeds_other, _, _ = _planned(sweep, capsys, ["--model", "qlstm"] + base[:4] + ["--n-seeds", "6", "--master-seed", "6"])
    assert seeds_other != seeds_a


def test_sweep_accepts_explicit_seeds_and_rejects_bad_options(tmp_path, capsys):
    sweep = load_script("run_sweep")
    seeds, dirs, _ = _planned(sweep, capsys, ["--model", "qslstm", "--scale", "pilot", "--seeds", "3", "1",
                                              "--save-dir", str(tmp_path)])
    assert seeds == [3, 1] and dirs == [("seed_3", "qslstm"), ("seed_1", "qslstm")]
    with pytest.raises(SystemExit):
        sweep.main(["--scale", "pilot", "--dry-run"])  # --model is required: no run-both-models default
    with pytest.raises(SystemExit):
        sweep.main(["--model", "qlstm", "--n-seeds", "2", "--seeds", "1", "2", "--dry-run"])
    with pytest.raises(SystemExit):
        sweep.main(["--model", "qlstm", "--workers", "0", "--dry-run"])
    args = sweep.build_parser().parse_args(["--model", "qslstm", "--scale", "pilot", "--run-extrapolation",
                                            "--jobs", "2"])
    assert args.workers == 2  # --jobs remains an alias of --workers
    argv = sweep.child_arguments(args, "qslstm", 3)
    assert argv[:4] == ["--model", "qslstm", "--seed", "3"] and "--run-extrapolation" in argv
    assert argv.count("--model") == 1
    assert not {"--workers", "--n-seeds", "--master-seed", "--dry-run"} & set(argv)


@pytest.mark.slow
def test_separately_run_models_combine_in_one_analysis(tmp_path):
    sweep = load_script("run_sweep")
    common = ["--scale", "pilot", "--sequence-length", "6", "--train-size", "12", "--val-fraction", "0.25",
              "--test-size", "4", "--hidden-size", "2", "--qnn-depth", "1", "--batch-size", "6", "--epochs", "1",
              "--n-seeds", "2", "--master-seed", "1", "--workers", "2", "--save-dir", str(tmp_path)]
    assert sweep.main(["--model", "qlstm"] + common) == 0
    analyze = load_script("analyze_results")
    root = tmp_path / "pilot-overridden"
    with pytest.raises(SystemExit, match="qslstm"):
        analyze.analyze(root, tmp_path / "early")  # only one model finished so far
    assert sweep.main(["--model", "qslstm"] + common) == 0
    assert sweep.main(["--model", "qslstm", "--resume"] + common) == 0  # finished runs are skipped
    out = analyze.analyze(root, tmp_path / "combined")
    table = pd.read_csv(out / "primary_paired_differences_test.csv")
    seeds = json.loads((root / "seeds_qlstm.json").read_text())["seeds"]
    assert sorted(table["run_seed"]) == sorted(seeds)
    assert all(c["dataset_checksums_equal"] and c["initial_shared_parameters_equal"]
               for c in json.loads((out / "pairing_check.json").read_text()).values())


def test_training_cli_exposes_required_arguments():
    train = load_script("train_nearest_neighbor")
    help_text = train.build_parser().format_help()
    for flag in ("--model", "--scale", "--sequence-length", "--train-size", "--test-size",
                 "--extrapolation-length", "--extrapolation-size", "--run-extrapolation", "--seed", "--data-seed",
                 "--hidden-size", "--qnn-depth", "--gate-epsilon", "--batch-size", "--epochs", "--lr",
                 "--weight-decay", "--record-margin", "--near-best-delta", "--value-separation",
                 "--device", "--save-dir"):
        assert flag in help_text, flag
