# Training / evaluation pipeline for the Ontario solar-generation forecasting experiment.
#
# Compares the conventional QLSTM and the stabilized Q-sLSTM, each behind the same Linear(13, 3) + tanh
# input projection, on next-hour generation. The model sees only `inputs` [B, L, 13]; the loss uses the
# final output step only. Every run trains the full epoch budget and keeps the lowest-validation-MSE
# checkpoint (earliest on ties), which is evaluated on the test split exactly once.

from __future__ import annotations

import copy
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from q_slstm.experiments.nearest_neighbor import environment_info, parameter_checksums
from q_slstm.models.factory import QUANTUM_MODELS, count_trainable_parameters
from q_slstm.models.projected_quantum import PROJECTION_ACTIVATION, build_projected_quantum_model
from q_slstm.models.q_slstm_cell import DEFAULT_GATE_EPSILON, QSLSTM_RECURRENCE
from q_slstm.models.q_slstm_log_cell import QSLSTM_LOG_RECURRENCE
from q_slstm.utils.seeds import epoch_permutation, stable_seed

from q_slstm.data_sources.cache import atomic_write_json, canonical_json, sha256_bytes, sha256_file
from q_slstm.datasets.solar_generation import RAW_INPUT_SIZE, PreparedDataset, unscale_target
from q_slstm.experiments import solar_generation_metrics as sm

REPO_ROOT = Path(__file__).resolve().parents[3]
ALPHA_TOLERANCE = 1e-5
HORIZON_HOURS = 1
OUTPUT_SIZE = 1

PRESETS = {
    "paper": dict(sequence_length=32, projection_size=3, hidden_size=6, qnn_depth=3, batch_size=64,
                  epochs=60, lr=1e-2, n_seeds=20),
    "pilot": dict(sequence_length=8, projection_size=3, hidden_size=2, qnn_depth=1, batch_size=8,
                  epochs=2, lr=1e-2, n_seeds=2),
}
TRAIN_FIELDS = ("projection_size", "hidden_size", "qnn_depth", "batch_size", "epochs", "lr")
SEED_TAGS = {"loader": 3, "model": 4, "projection": 5}
REQUIRED_ARTIFACTS = (
    "config.json", "environment.json", "source_revision.json", "dataset_manifest.json", "scaler.json",
    "parameters.json", "init_parameters.json", "history.csv", "checkpoints/best.pt", "checkpoints/last.pt",
    "train_summary.json", "predictions_test.csv", "per_day_metrics_test.csv", "per_seed_metrics_test.csv",
    "alpha_metrics_test.csv", "timing.json",
)
PREDICTION_COLUMNS = (
    "run_seed", "model", "dataset_id", "window_id", "origin_timestamp_utc", "target_timestamp_utc",
    "horizon_hours", "target_mwh", "prediction_mwh", "absolute_error", "squared_error", "persistence_mwh",
    "daily_persistence_mwh", "target_month", "target_hour_est", "is_daylight_proxy", "alpha_final_mean",
)


class NonFiniteError(RuntimeError):
    """Raised with run/epoch/batch identity when inputs, outputs, loss, or gradients are non-finite."""


class IncompatibleRunError(RuntimeError):
    """A run directory holds a run with a different configuration or dataset."""


# ---------------------------------------------------------------------------------------------
# Arguments, presets, seeds, identities
# ---------------------------------------------------------------------------------------------

def add_run_arguments(parser):
    """Arguments shared by the training script and the sweep launcher."""
    a = parser.add_argument
    a("--scale", choices=sorted(PRESETS), default="paper")
    a("--dataset-manifest", default=None,
      help="prepared_<scale>.json or a processed dataset_manifest.json (default: "
           "data/solar_generation/prepared_<scale>.json)")
    a("--projection-size", type=int, default=None, help="quantum input width after Linear(13, k)+tanh (3)")
    a("--hidden-size", type=int, default=None)
    a("--qnn-depth", type=int, default=None)
    a("--gate-epsilon", type=float, default=DEFAULT_GATE_EPSILON, help="legacy compatibility setting; polynomial qslstm has no gate clipping or denominator floor")
    a("--batch-size", type=int, default=None)
    a("--epochs", type=int, default=None)
    a("--lr", type=float, default=None)
    a("--weight-decay", type=float, default=0.0)
    a("--grad-clip", type=float, default=0.0, help="max gradient norm; 0 disables (norms are still logged)")
    a("--horizon", type=int, default=None, help=f"fixed to {HORIZON_HOURS}; other values are rejected")
    a("--raw-input-size", type=int, default=None, help=f"fixed to {RAW_INPUT_SIZE}; other values are rejected")
    a("--save-alpha-trace", action="store_true", help="also save the per-input-step alpha trace per window")
    a("--device", choices=["cpu", "cuda"], default="cpu")
    a("--save-dir", default="results/solar_generation")


def derive_seeds(run_seed):
    """Independent model / projection / loader seeds from the run seed. Data do not depend on seeds."""
    return {
        "run_seed": int(run_seed),
        "model": stable_seed(run_seed, SEED_TAGS["model"]),
        "projection": stable_seed(run_seed, SEED_TAGS["projection"]),
        "loader": stable_seed(run_seed, SEED_TAGS["loader"]),
        "data_seed": None,
        "data_seed_note": "inapplicable: one fixed dataset and chronological split for every seed and model",
        "epoch_order_rule": "numpy default_rng(SeedSequence([loader, epoch])).permutation(n_train_windows)",
    }


def resolve_manifest_path(path, scale):
    return Path(path) if path else Path("data/solar_generation") / f"prepared_{scale}.json"


def _read_dataset_manifest(path):
    path = Path(path)
    if path.name.startswith("prepared_"):
        pointer = json.loads(path.read_text(encoding="utf-8"))
        path = path.parent / pointer["manifest"]
    return json.loads(path.read_text(encoding="utf-8"))


def short_hash(obj, n=12):
    return sha256_bytes(canonical_json(obj).encode("utf-8"))[:n]


def resolve_config(args):
    """Merge preset, dataset, and explicit overrides into one validated config dict."""
    a = dict(vars(args)) if not isinstance(args, dict) else dict(args)
    if a.get("model") not in QUANTUM_MODELS:
        raise ValueError(f"--model must be one of {QUANTUM_MODELS}, got {a.get('model')!r}")
    if a.get("horizon") not in (None, HORIZON_HOURS):
        raise ValueError(f"horizon is fixed to {HORIZON_HOURS} hour for this experiment, got {a['horizon']}")
    if a.get("raw_input_size") not in (None, RAW_INPUT_SIZE):
        raise ValueError(f"raw input width is fixed to {RAW_INPUT_SIZE}, got {a['raw_input_size']}")
    scale = a.get("scale") or "paper"
    if scale not in PRESETS:
        raise ValueError(f"unknown scale preset {scale!r}")
    manifest_path = resolve_manifest_path(a.get("dataset_manifest"), scale)
    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_path} not found; run prepare_dataset.py --scale {scale} first")
    dataset = _read_dataset_manifest(manifest_path)
    dataset_scale = dataset["identity"]["preparation"]["scale"]
    if dataset_scale != scale:
        raise ValueError(f"dataset {dataset['dataset_id']} was prepared for scale {dataset_scale!r}, "
                         f"not {scale!r}; prepare a matching dataset")
    if dataset["raw_input_size"] != RAW_INPUT_SIZE or dataset["target"]["horizon_hours"] != HORIZON_HOURS:
        raise ValueError("dataset does not have 13 input columns and a one-hour horizon")

    preset = dict(PRESETS[scale])
    overrides = {}
    for key in TRAIN_FIELDS:
        if a.get(key) is not None and a[key] != preset[key]:
            overrides[key] = {"preset": preset[key], "used": a[key]}
            preset[key] = a[key]
    preset.pop("n_seeds")
    preset_length = preset.pop("sequence_length")
    for key in ("weight_decay", "grad_clip"):
        if a.get(key):
            overrides[key] = {"preset": 0.0, "used": a[key]}
    dataset_overrides = dataset["identity"]["preparation"]["overrides"]
    if dataset["sequence_length"] != preset_length and "sequence_length" not in dataset_overrides:
        raise ValueError("dataset sequence length conflicts with the scale preset")
    for key in ("epochs", "batch_size", "hidden_size", "qnn_depth", "projection_size"):
        if preset[key] < 1:
            raise ValueError(f"{key} must be positive, got {preset[key]}")
    if not preset["lr"] > 0:
        raise ValueError("lr must be positive")
    labeled = bool(overrides or dataset_overrides)

    study = {
        "kind": "solar_generation_study",
        "dataset_id": dataset["dataset_id"],
        "scale_preset": scale,
        "scale_label": scale if not labeled else f"{scale}-overridden",
        "training_overrides": overrides,
        "dataset_overrides": dataset_overrides,
        "sequence_length": dataset["sequence_length"],
        "horizon_hours": HORIZON_HOURS,
        "raw_input_size": RAW_INPUT_SIZE,
        "quantum_input_size": preset["projection_size"],
        "projection": f"Linear({RAW_INPUT_SIZE}, {preset['projection_size']}) + {PROJECTION_ACTIVATION} at every timestep",
        "projection_activation": PROJECTION_ACTIVATION,
        **preset,
        "output_size": OUTPUT_SIZE,
        "n_qubits": preset["projection_size"] + preset["hidden_size"],
        "n_vqcs_per_cell": 4,
        "gate_epsilon": a.get("gate_epsilon", DEFAULT_GATE_EPSILON),
        "qslstm_recurrence": QSLSTM_LOG_RECURRENCE if a["model"] == "qslstm_log" else QSLSTM_RECURRENCE,
        "weight_decay": a.get("weight_decay", 0.0) or 0.0,
        "grad_clip": a.get("grad_clip", 0.0) or 0.0,
        "optimizer": "Adam",
        "device": a.get("device", "cpu"),
        "precision": "float32",
        "simulator": "pennylane default.qubit (parameter broadcasting)",
        "save_alpha_trace": bool(a.get("save_alpha_trace", False)),
        "loss": "MSE on the standardized final output step only (outputs[:, -1, :] vs target [B, 1])",
        "selection": "lowest validation MSE (earliest epoch on exact ties); no early stopping",
    }
    study_id = short_hash(study)
    run_seed = int(a.get("seed", 0))
    config = {
        "kind": "solar_generation_run",
        "model": a["model"],
        "study_id": study_id,
        "study": study,
        "seeds": derive_seeds(run_seed),
        "dataset_manifest": str(manifest_path),
        "save_dir": str(a.get("save_dir") or "results/solar_generation"),
    }
    config["config_hash"] = short_hash({k: v for k, v in config.items() if k not in ("dataset_manifest", "save_dir")}, 16)
    return config


def study_directory(config):
    return Path(config["save_dir"]) / config["study"]["scale_label"] / config["study_id"]


def run_directory(config):
    return study_directory(config) / f"seed_{config['seeds']['run_seed']}" / config["model"]


# ---------------------------------------------------------------------------------------------
# Batching, loss, finite checks
# ---------------------------------------------------------------------------------------------

def batch_tensors(dataset, positions):
    """inputs [B, L, 13] and target [B, 1] for dataset positions (model inputs never include metadata)."""
    L = dataset.sequence_length
    idx = dataset.target_index[positions]
    inputs = torch.stack([dataset.features[j - L:j] for j in idx])
    target = dataset.targets[torch.as_tensor(idx)].unsqueeze(1)
    return inputs, target


def final_step_mse(outputs, target):
    """MSE between the final output step [B, 1] and the target [B, 1]; earlier steps are ignored."""
    pred = outputs[:, -1, :]
    if not (pred.shape == target.shape and pred.ndim == 2 and pred.shape[1] == 1):
        raise ValueError(f"prediction {tuple(pred.shape)} and target {tuple(target.shape)} must both be [B, 1]")
    return ((pred - target) ** 2).mean()


def _require_finite(tensor, what, context):
    if not torch.isfinite(tensor).all():
        raise NonFiniteError(f"non-finite {what} at {context}")


def _context(config, epoch, batch):
    return f"run_seed={config['seeds']['run_seed']} model={config['model']} epoch={epoch} batch={batch}"


def _batches(n, batch_size, order=None):
    order = np.arange(n) if order is None else order
    return [order[i:i + batch_size] for i in range(0, n, batch_size)]


@torch.no_grad()
def evaluate_mse(model, dataset, config, epoch):
    """Sample-weighted standardized MSE of the final step over a split."""
    model.eval()
    total, n = 0.0, 0
    for b, pos in enumerate(_batches(len(dataset), config["study"]["batch_size"])):
        inputs, target = batch_tensors(dataset, pos)
        outputs = model(inputs.to(config["study"]["device"]))[0]
        _require_finite(outputs, "validation outputs", _context(config, epoch, f"val{b}"))
        pred = outputs[:, -1, :]
        total += float(((pred - target.to(pred.device)) ** 2).sum())
        n += len(pos)
    return total / n


# ---------------------------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------------------------

def train_model(model, train_ds, val_ds, config, run_dir, resume_state=None):
    """Adam on final-step MSE for the full epoch budget; saves best.pt (validation) and last.pt."""
    study, device = config["study"], config["study"]["device"]
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=study["lr"], weight_decay=study["weight_decay"])
    clip = study["grad_clip"]
    target_scale = json.loads((run_dir / "scaler.json").read_text(encoding="utf-8"))["target"]["scale"]
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    history, best_val, best_epoch, start_epoch = [], math.inf, 0, 1
    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        history, best_val, best_epoch = resume_state["history"], resume_state["best_val_mse"], resume_state["best_epoch"]
        start_epoch = resume_state["epoch"] + 1
        torch.set_rng_state(resume_state["torch_rng_state"])

    for epoch in range(start_epoch, study["epochs"] + 1):
        model.train()
        start = time.perf_counter()
        order = epoch_permutation(config["seeds"]["loader"], epoch, len(train_ds))
        order_hash = sha256_bytes(train_ds.window_ids[order].astype("<i8").tobytes())
        sq_sum, n_seen, norms, clipped = 0.0, 0, [], 0
        for b, pos in enumerate(_batches(len(train_ds), study["batch_size"], order)):
            context = _context(config, epoch, b)
            inputs, target = batch_tensors(train_ds, pos)
            inputs, target = inputs.to(device), target.to(device)
            _require_finite(inputs, "inputs", context)
            optimizer.zero_grad()
            outputs = model(inputs)[0]
            _require_finite(outputs, "outputs", context)
            loss = final_step_mse(outputs, target)
            _require_finite(loss, "loss", context)
            loss.backward()
            for name, p in model.named_parameters():
                if p.grad is not None:
                    _require_finite(p.grad, f"gradient of {name}", context)
            norm = float(torch.nn.utils.clip_grad_norm_(params, max_norm=clip if clip > 0 else float("inf")))
            clipped += int(clip > 0 and norm > clip)
            optimizer.step()
            sq_sum += loss.item() * len(pos)  # sample-weighted, including the partial final batch
            n_seen += len(pos)
            norms.append(norm)

        val_mse = evaluate_mse(model, val_ds, config, epoch)
        improved = val_mse < best_val
        row = {
            "epoch": epoch, "train_mse_scaled": sq_sum / n_seen, "val_mse_scaled": val_mse,
            "train_mse_mwh2": sq_sum / n_seen * target_scale**2, "val_mse_mwh2": val_mse * target_scale**2,
            "grad_norm_mean": float(np.mean(norms)), "grad_norm_max": float(np.max(norms)),
            "clipped_batches": clipped, "n_batches": len(norms), "clip_fraction": clipped / len(norms),
            "n_train_windows": n_seen, "epoch_order_sha256": order_hash,
            "epoch_seconds": time.perf_counter() - start, "is_best": bool(improved),
        }
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        print(f"[{config['model']} seed {config['seeds']['run_seed']}] epoch {epoch:03d} "
              f"train {row['train_mse_scaled']:.5f} val {val_mse:.5f} (scaled) "
              f"gnorm {row['grad_norm_mean']:.3g} ({row['epoch_seconds']:.1f}s)" + (" *" if improved else ""),
              flush=True)

        payload = {"epoch": epoch, "model_state_dict": copy.deepcopy(model.state_dict()), "val_mse_scaled": val_mse,
                   "config": config, "config_hash": config["config_hash"], "dataset_id": config["study"]["dataset_id"],
                   "scaler": json.loads((run_dir / "scaler.json").read_text(encoding="utf-8"))}
        if improved:
            best_val, best_epoch = val_mse, epoch
            _atomic_torch_save(payload, ckpt_dir / "best.pt")
        _atomic_torch_save({**payload, "optimizer_state_dict": optimizer.state_dict(),
                            "torch_rng_state": torch.get_rng_state(), "history": history,
                            "best_val_mse": best_val, "best_epoch": best_epoch}, ckpt_dir / "last.pt")

    return {
        "best_epoch": best_epoch, "best_val_mse_scaled": best_val, "best_val_mse_mwh2": best_val * target_scale**2,
        "epochs_run": len(history), "early_stopping": "none",
        "epoch_order_sha256": sha256_bytes("".join(r["epoch_order_sha256"] for r in history).encode()),
        "epoch1_order_sha256": history[0]["epoch_order_sha256"],
        "resumed_from_epoch": None if resume_state is None else resume_state["epoch"],
    }


def _atomic_torch_save(obj, path):
    tmp = path.with_name(path.name + ".partial")
    torch.save(obj, tmp)
    tmp.replace(path)


# ---------------------------------------------------------------------------------------------
# Evaluation (held-out test, once)
# ---------------------------------------------------------------------------------------------

@torch.no_grad()
def predict_with_diagnostics(model, dataset, config):
    """Final-step predictions (scaled) and hidden-mean alpha per input step [n, L], in dataset order."""
    model.eval()
    preds, alphas = [], []
    for b, pos in enumerate(_batches(len(dataset), config["study"]["batch_size"])):
        inputs, _ = batch_tensors(dataset, pos)
        outputs, _, diag = model(inputs.to(config["study"]["device"]), return_diagnostics=True)
        alpha = diag["alpha"]  # [B, L, hidden], elementwise ratio
        context = f"test batch {b}"
        _require_finite(outputs, "test outputs", context)
        _require_finite(alpha, "alpha", context)
        if (alpha < -ALPHA_TOLERANCE).any() or (alpha > 1 + ALPHA_TOLERANCE).any():
            raise ValueError(f"alpha outside [0, 1] at {context}")
        preds.append(outputs[:, -1, 0].cpu())
        alphas.append(alpha.mean(dim=-1).cpu())  # reduce over hidden units only after the ratio
    return torch.cat(preds).numpy().astype(np.float64), torch.cat(alphas).numpy().astype(np.float64)


def prediction_frame(dataset, pred_scaled, alpha_steps, scaler, config):
    w = dataset.windows
    pred = unscale_target(pred_scaled, scaler)
    target = w["target_mwh"].to_numpy(dtype=float)
    err = pred - target
    frame = pd.DataFrame({
        "run_seed": config["seeds"]["run_seed"], "model": config["model"], "dataset_id": config["study"]["dataset_id"],
        "window_id": w["window_id"].to_numpy(), "origin_timestamp_utc": w["origin_timestamp_utc"].to_numpy(),
        "target_timestamp_utc": w["target_timestamp_utc"].to_numpy(), "horizon_hours": HORIZON_HOURS,
        "target_mwh": target, "prediction_mwh": pred, "absolute_error": np.abs(err), "squared_error": err**2,
        "persistence_mwh": w["persistence_mwh"].to_numpy(dtype=float),
        "daily_persistence_mwh": w["daily_persistence_mwh"].to_numpy(dtype=float),
        "target_month": w["target_month"].to_numpy(), "target_hour_est": w["target_hour_est"].to_numpy(),
        "is_daylight_proxy": w["is_daylight_proxy"].to_numpy(), "alpha_final_mean": alpha_steps[:, -1],
    })
    frame["split"] = "test"
    frame["target_date_est"] = w["target_date_est"].to_numpy()
    if "is_large_ramp" in w:
        frame["abs_ramp_mwh"] = w["abs_ramp_mwh"].to_numpy(dtype=float)
        frame["is_large_ramp"] = w["is_large_ramp"].to_numpy(dtype=bool)
    return frame


def evaluate_test(model, data, config, run_dir):
    test_ds = data.torch_dataset("test")
    pred_scaled, alpha_steps = predict_with_diagnostics(model, test_ds, config)
    frame = prediction_frame(test_ds, pred_scaled, alpha_steps, data.scaler, config)
    frame.to_csv(run_dir / "predictions_test.csv", index=False)
    per_seed = sm.seed_metrics(frame)
    per_seed.to_csv(run_dir / "per_seed_metrics_test.csv", index=False)
    sm.daily_metrics(frame).to_csv(run_dir / "per_day_metrics_test.csv", index=False)
    identity = {"run_seed": config["seeds"]["run_seed"], "model": config["model"]}
    sm.alpha_curves(alpha_steps, test_ds.windows["is_daylight_proxy"], identity).to_csv(
        run_dir / "alpha_metrics_test.csv", index=False)
    if config["study"]["save_alpha_trace"]:
        L = test_ds.sequence_length
        ts = data.table["timestamp_utc"].to_numpy()
        steps = np.arange(1, L)  # step 0 carries no information (zero-initialized normalizer)
        trace = pd.DataFrame({
            "window_id": np.repeat(test_ds.window_ids, len(steps)),
            "input_step": np.tile(steps, len(test_ds)),
            "input_timestamp_utc": ts[(test_ds.target_index[:, None] - L + steps[None, :]).ravel()],
            "alpha_mean": alpha_steps[:, 1:].ravel(),
        })
        trace.to_csv(run_dir / "alpha_trace_test.csv", index=False)
    return per_seed


# ---------------------------------------------------------------------------------------------
# Provenance, status, completion
# ---------------------------------------------------------------------------------------------

def source_revision():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                         stderr=subprocess.DEVNULL).decode().strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT,
                                         stderr=subprocess.DEVNULL).decode()
        return {"kind": "git", "commit": commit, "dirty": bool(status.strip()),
                "dirty_paths": [line[3:] for line in status.splitlines()][:200]}
    except Exception:
        return {"kind": "unavailable", "commit": None, "dirty": None}


def _write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def artifact_hashes(run_dir, names=REQUIRED_ARTIFACTS):
    return {name: sha256_file(Path(run_dir) / name) for name in names}


def run_status(config, run_dir=None):
    """'new', 'complete', 'incomplete', or 'incompatible' for the directory of `config`."""
    run_dir = Path(run_dir) if run_dir is not None else run_directory(config)
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        return "new"
    existing = json.loads(cfg_path.read_text(encoding="utf-8"))
    if existing.get("config_hash") != config["config_hash"]:
        return "incompatible"
    done = run_dir / "complete.json"
    if not done.exists():
        return "incomplete"
    record = json.loads(done.read_text(encoding="utf-8"))
    if record.get("config_hash") != config["config_hash"] or record.get("dataset_id") != config["study"]["dataset_id"]:
        return "incompatible"
    try:
        if artifact_hashes(run_dir, record["artifact_sha256"].keys()) != record["artifact_sha256"]:
            return "incomplete"
    except FileNotFoundError:
        return "incomplete"
    return "complete"


def _resume_state(config, run_dir):
    last = run_dir / "checkpoints" / "last.pt"
    if not last.exists():
        return None
    state = torch.load(last, map_location="cpu", weights_only=False)
    if state.get("config_hash") != config["config_hash"] or state.get("dataset_id") != config["study"]["dataset_id"]:
        return None
    return state


def run_experiment(config, run_dir=None, resume=False, overwrite=False):
    """Train one model on one seed, evaluate its best-validation checkpoint once, and write artifacts.

    `resume`: skip a verified complete run, or continue an interrupted one from last.pt.
    `overwrite`: rerun even when a complete run exists. Incompatible directories are refused.
    """
    run_dir = Path(run_dir) if run_dir is not None else run_directory(config)
    status = run_status(config, run_dir)
    if status == "incompatible" and not overwrite:
        raise IncompatibleRunError(f"{run_dir} holds a run with a different configuration or dataset; "
                                   "use a different --save-dir or remove it")
    if status == "complete":
        if resume:
            print(f"verified complete run, skipping: {run_dir}", flush=True)
            return run_dir
        if not overwrite:
            raise IncompatibleRunError(f"{run_dir} is already complete; pass --resume to skip or --overwrite to rerun")
    resume_state = _resume_state(config, run_dir) if (resume and status == "incomplete") else None
    if resume_state is None and run_dir.exists():
        for child in run_dir.iterdir():  # fresh start: clear stale artifacts, keep the live console log
            if child.name != "console_log.txt":
                shutil.rmtree(child) if child.is_dir() else child.unlink()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "complete.json").unlink(missing_ok=True)
    (run_dir / "failure.json").unlink(missing_ok=True)

    started = time.perf_counter()
    timing = {}
    try:
        _write_json(run_dir / "config.json", config)
        env = environment_info()
        env.update(device=config["study"]["device"], simulator=config["study"]["simulator"],
                   dtype=config["study"]["precision"], argv=sys.argv)
        _write_json(run_dir / "environment.json", env)
        _write_json(run_dir / "source_revision.json", source_revision())

        t0 = time.perf_counter()
        data = PreparedDataset(config["dataset_manifest"])
        if data.dataset_id != config["study"]["dataset_id"]:
            raise IncompatibleRunError(f"dataset {data.dataset_id} differs from configured {config['study']['dataset_id']}")
        shutil.copyfile(data.dir / "dataset_manifest.json", run_dir / "dataset_manifest.json")
        shutil.copyfile(data.dir / "scaler.json", run_dir / "scaler.json")
        train_ds, val_ds = data.torch_dataset("train"), data.torch_dataset("val")
        timing["data_seconds"] = time.perf_counter() - t0

        s = config["study"]
        model = build_projected_quantum_model(
            config["model"], RAW_INPUT_SIZE, s["projection_size"], s["hidden_size"], OUTPUT_SIZE, s["qnn_depth"],
            gate_epsilon=s["gate_epsilon"], device=s["device"], model_seed=config["seeds"]["model"],
            projection_seed=config["seeds"]["projection"])
        n_params = count_trainable_parameters(model)
        _write_json(run_dir / "parameters.json", {
            "trainable_parameters": n_params,
            "projection_parameters": sum(p.numel() for p in model.projection.parameters()),
            "raw_input_size": RAW_INPUT_SIZE, "quantum_input_size": s["projection_size"],
            "projection_activation": PROJECTION_ACTIVATION, "n_qubits": model.n_qubits,
            "hidden_size": s["hidden_size"], "qnn_depth": s["qnn_depth"],
            "parameter_shapes": {k: list(v.shape) for k, v in model.state_dict().items()},
        })
        _write_json(run_dir / "init_parameters.json", {
            "model_seed": config["seeds"]["model"], "projection_seed": config["seeds"]["projection"],
            "checksums": parameter_checksums(model),
            "note": "sha256 of every parameter tensor (projection included) before training",
        })
        if resume_state is not None:
            model.load_state_dict(resume_state["model_state_dict"])
            print(f"resuming from epoch {resume_state['epoch']} ({run_dir / 'checkpoints' / 'last.pt'})", flush=True)

        t0 = time.perf_counter()
        result = train_model(model, train_ds, val_ds, config, run_dir, resume_state)
        timing["train_seconds"] = time.perf_counter() - t0
        result.update(n_train_windows=len(train_ds), n_val_windows=len(val_ds))
        _write_json(run_dir / "train_summary.json", result)

        # Test data are touched only now, once, with the checkpoint selected on validation.
        best = torch.load(run_dir / "checkpoints" / "best.pt", map_location=s["device"], weights_only=False)
        model.load_state_dict(best["model_state_dict"])
        t0 = time.perf_counter()
        per_seed = evaluate_test(model, data, config, run_dir)
        timing["evaluation_seconds"] = time.perf_counter() - t0
        timing["total_seconds"] = time.perf_counter() - started
        _write_json(run_dir / "timing.json", timing)

        missing = [n for n in REQUIRED_ARTIFACTS if not (run_dir / n).exists()]
        if missing:
            raise RuntimeError(f"required artifacts missing: {missing}")
        preds = pd.read_csv(run_dir / "predictions_test.csv")
        if list(preds.columns[:len(PREDICTION_COLUMNS)]) != list(PREDICTION_COLUMNS) or len(preds) != len(data.split_windows("test")):
            raise RuntimeError("predictions_test.csv failed validation")
        overall = per_seed.query("subset == 'all_eligible' and group_type == 'overall' and forecaster == 'model'").iloc[0]
        _write_json(run_dir / "complete.json", {
            "model": config["model"], "run_seed": config["seeds"]["run_seed"], "study_id": config["study_id"],
            "config_hash": config["config_hash"], "dataset_id": config["study"]["dataset_id"],
            "scale_label": s["scale_label"], "best_epoch": result["best_epoch"],
            "test_mse_mwh2": float(overall["mse"]), "n_test_windows": int(overall["n"]),
            "trainable_parameters": n_params,
            "test_window_ids_sha256": sha256_bytes(preds["window_id"].to_numpy(dtype="<i8").tobytes()),
            "artifact_sha256": artifact_hashes(run_dir),
        })
        return run_dir
    except BaseException as exc:
        _write_json(run_dir / "failure.json", {
            "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(),
            "model": config["model"], "run_seed": config["seeds"]["run_seed"],
            "elapsed_seconds": time.perf_counter() - started,
        })
        raise


def verify_pairing(run_dir_a, run_dir_b):
    """Checks that two runs of one seed differ only in the recurrent architecture."""
    load = lambda d, f: json.loads((Path(d) / f).read_text(encoding="utf-8"))
    ca, cb = load(run_dir_a, "config.json"), load(run_dir_b, "config.json")
    ia, ib = load(run_dir_a, "init_parameters.json"), load(run_dir_b, "init_parameters.json")
    sa, sb = load(run_dir_a, "train_summary.json"), load(run_dir_b, "train_summary.json")
    pa, pb = load(run_dir_a, "parameters.json"), load(run_dir_b, "parameters.json")
    da, db = load(run_dir_a, "complete.json"), load(run_dir_b, "complete.json")
    names_equal = set(ia["checksums"]) == set(ib["checksums"])
    checks = {
        "study_settings_equal": ca["study"] == cb["study"],
        "study_id_equal": ca["study_id"] == cb["study_id"],
        "dataset_id_equal": ca["study"]["dataset_id"] == cb["study"]["dataset_id"],
        "seed_map_equal": ca["seeds"] == cb["seeds"],
        "scaler_equal": sha256_file(Path(run_dir_a) / "scaler.json") == sha256_file(Path(run_dir_b) / "scaler.json"),
        "epoch_orders_equal": sa["epoch_order_sha256"] == sb["epoch_order_sha256"],
        "all_parameter_names_shared": names_equal,
        "initial_parameters_equal": names_equal and all(ia["checksums"][k] == ib["checksums"][k] for k in ia["checksums"]),
        "parameter_counts_equal": pa["trainable_parameters"] == pb["trainable_parameters"],
        "test_target_sets_equal": da["test_window_ids_sha256"] == db["test_window_ids_sha256"],
        "models_differ": ca["model"] != cb["model"],
    }
    checks["accepted"] = all(checks.values())
    return checks
