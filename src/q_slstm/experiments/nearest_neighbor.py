# Training / evaluation pipeline for the nearest-neighbor memory-revision experiment.
#
# Compares the conventional QLSTM and the stabilized Q-sLSTM (same four-VQC architecture) on the
# synthetic task in q_slstm.datasets.nearest_neighbor. Models see only `inputs`; masks, similarities,
# best indices, and case labels are used for the loss mask, metrics, and post-hoc analysis.

from __future__ import annotations

import copy
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from q_slstm.datasets.nearest_neighbor import (
    CASE_TYPES,
    INPUT_SIZE,
    OUTPUT_SIZE,
    NearestNeighborConfig,
    make_balanced_suite,
    make_train_pool,
    split_train_val,
)
from q_slstm.experiments import nearest_neighbor_metrics as nnm
from q_slstm.models.factory import QUANTUM_MODELS, build_quantum_model, count_trainable_parameters
from q_slstm.models.q_slstm_cell import DEFAULT_GATE_EPSILON, QSLSTM_RECURRENCE
from q_slstm.models.q_slstm_log_cell import QSLSTM_LOG_RECURRENCE
from q_slstm.utils.seeds import stable_seed

REPO_ROOT = Path(__file__).resolve().parents[3]
ALPHA_TOLERANCE = 1e-5

# Split: the paper preset uses 5,000 sequences in total, split 70% / 10% / 20% into optimizer-training
# (3,500), validation (500), and held-out test (1,000). The training pool below is train + validation.
# Scale-controlling settings. "paper" is the reported study; "pilot" is a small pipeline / VQC-cost
# check whose artifacts are always labeled as such. Explicit CLI values override either preset and
# are recorded (the scale label then says so).
PRESETS = {
    "paper": dict(
        sequence_length=32, train_size=4000, val_fraction=0.125, test_size=1000,
        extrapolation_length=64, extrapolation_size=250, hidden_size=6, qnn_depth=3,
        batch_size=64, epochs=100, lr=1e-2, n_seeds=5,
    ),
    "pilot": dict(
        sequence_length=8, train_size=32, val_fraction=0.125, test_size=8,
        extrapolation_length=16, extrapolation_size=4, hidden_size=2, qnn_depth=1,
        batch_size=8, epochs=2, lr=1e-2, n_seeds=2,
    ),
}
PRESET_FIELDS = tuple(k for k in PRESETS["paper"] if k != "n_seeds")


class NonFiniteError(RuntimeError):
    """Raised with run/epoch/batch identity when inputs, outputs, loss, or gradients are non-finite."""


# ---------------------------------------------------------------------------------------------
# Arguments, presets, seeds
# ---------------------------------------------------------------------------------------------

def add_run_arguments(parser):
    """Arguments shared by the training script and the sweep launcher (all optional here)."""
    a = parser.add_argument
    a("--scale", choices=sorted(PRESETS), default="paper", help="scale preset (default: paper)")
    a("--sequence-length", type=int, default=None, help="total tokens incl. the reference token (paper: 32)")
    a("--train-size", type=int, default=None,
      help="IID pool size = training + validation (paper: 4000, split 3500 / 500)")
    a("--val-fraction", type=float, default=None,
      help="fraction of the pool used for validation (paper: 0.125, i.e. 10%% of train+val+test)")
    a("--test-size", type=int, default=None, help="balanced held-out suite size (paper: 1000)")
    a("--extrapolation-length", type=int, default=None, help="total tokens of the extrapolation suite (paper: 64)")
    a("--extrapolation-size", type=int, default=None, help="extrapolation suite size (paper: 250)")
    a("--run-extrapolation", action="store_true", help="also evaluate the length-trained checkpoint at the extrapolation length")
    a("--data-seed", type=int, default=None, help="base seed for data and split (default: derived from --seed)")
    a("--hidden-size", type=int, default=None)
    a("--qnn-depth", type=int, default=None)
    a("--input-size", type=int, default=None, help=f"fixed to {INPUT_SIZE}; conflicting values are rejected")
    a("--output-size", type=int, default=None, help=f"fixed to {OUTPUT_SIZE}; conflicting values are rejected")
    a("--gate-epsilon", type=float, default=DEFAULT_GATE_EPSILON, help="legacy compatibility setting; polynomial qslstm has no gate clipping or denominator floor")
    a("--batch-size", type=int, default=None)
    a("--epochs", type=int, default=None)
    a("--lr", type=float, default=None, help="overrides the preset learning rate")
    a("--weight-decay", type=float, default=0.0)
    a("--grad-clip", type=float, default=0.0, help="max gradient norm; 0 disables clipping (norms are still logged)")
    a("--record-margin", type=float, default=NearestNeighborConfig.record_margin)
    a("--near-best-delta", type=float, default=NearestNeighborConfig.near_best_delta)
    a("--value-separation", type=float, default=NearestNeighborConfig.value_separation)
    a("--device", choices=["cpu", "cuda"], default="cpu")
    a("--save-dir", type=str, default="results/nearest_neighbor")


SEED_TAGS = {"data": 1, "split": 2, "loader": 3, "model": 4}


def derive_seeds(seed, data_seed=None):
    """Documented seed map. Models paired on one run seed share every entry.

    data, split : derived from `data_seed` (or the run seed when it is not given)
    loader      : train-loader shuffling, derived from the run seed
    model       : VQC / output-layer initialization, derived from the run seed
    """
    base = seed if data_seed is None else data_seed
    return {
        "run_seed": int(seed),
        "data_seed_base": int(base),
        "data": stable_seed(base, SEED_TAGS["data"]),
        "split": stable_seed(base, SEED_TAGS["split"]),
        "loader": stable_seed(seed, SEED_TAGS["loader"]),
        "model": stable_seed(seed, SEED_TAGS["model"]),
    }


def resolve_config(args):
    """Merge the preset with explicit overrides into one fully resolved, validated config dict."""
    a = dict(vars(args)) if not isinstance(args, dict) else dict(args)
    if a.get("model") not in QUANTUM_MODELS:
        raise ValueError(f"--model must be one of {QUANTUM_MODELS}, got {a.get('model')!r}")
    if a.get("input_size") not in (None, INPUT_SIZE):
        raise ValueError(f"input_size is fixed to {INPUT_SIZE} for this task, got {a['input_size']}")
    if a.get("output_size") not in (None, OUTPUT_SIZE):
        raise ValueError(f"output_size is fixed to {OUTPUT_SIZE} for this task, got {a['output_size']}")

    preset = a.get("scale", "paper")
    if preset not in PRESETS:
        raise ValueError(f"unknown scale preset {preset!r}")
    resolved = dict(PRESETS[preset])
    overrides = {}
    for key in PRESET_FIELDS:
        if a.get(key) is not None and a[key] != resolved[key]:
            overrides[key] = {"preset": resolved[key], "used": a[key]}
            resolved[key] = a[key]
    resolved.pop("n_seeds")

    generation = NearestNeighborConfig(
        record_margin=a.get("record_margin", NearestNeighborConfig.record_margin),
        near_best_delta=a.get("near_best_delta", NearestNeighborConfig.near_best_delta),
        value_separation=a.get("value_separation", NearestNeighborConfig.value_separation),
    )
    generation.validate(resolved["sequence_length"])

    val_size = int(round(resolved["train_size"] * resolved["val_fraction"]))
    if not 0 < val_size < resolved["train_size"]:
        raise ValueError("train_size and val_fraction leave no training or validation sequences")
    run_seed = int(a.get("seed", 0))
    config = {
        "kind": "nearest_neighbor_run",
        "model": a["model"],
        "scale_preset": preset,
        "scale_label": preset if not overrides else f"{preset}-overridden",
        "preset_overrides": overrides,
        **resolved,
        "n_candidates": resolved["sequence_length"] - 1,
        "val_size": val_size,
        "optimizer_train_size": resolved["train_size"] - val_size,
        "split_fractions": {  # of train + validation + held-out test
            name: n / (resolved["train_size"] + resolved["test_size"])
            for name, n in (("train", resolved["train_size"] - val_size), ("validation", val_size),
                            ("test", resolved["test_size"]))
        },
        "early_stopping": "none: every run trains for the full epoch budget; best-validation checkpoint kept",
        "run_extrapolation": bool(a.get("run_extrapolation", False)),
        "input_size": INPUT_SIZE,
        "output_size": OUTPUT_SIZE,
        "input_projection": False,
        "n_qubits": INPUT_SIZE + resolved["hidden_size"],
        "gate_epsilon": a.get("gate_epsilon", DEFAULT_GATE_EPSILON),
        "qslstm_recurrence": QSLSTM_LOG_RECURRENCE if a["model"] == "qslstm_log" else QSLSTM_RECURRENCE,
        "weight_decay": a.get("weight_decay", 0.0),
        "grad_clip": a.get("grad_clip", 0.0),
        "device": a.get("device", "cpu"),
        "generation": generation.to_dict(),
        "seeds": derive_seeds(run_seed, a.get("data_seed")),
        "save_dir": str(a.get("save_dir", "results/nearest_neighbor")),
        "loss": "mean squared error over loss_mask (every candidate, reference token excluded)",
        "selection_metric": "validation MSE over metric_mask (second candidate onward)",
    }
    for key in ("epochs", "batch_size"):
        if config[key] < 1:
            raise ValueError(f"{key} must be positive, got {config[key]}")
    if not config["lr"] > 0:
        raise ValueError(f"lr must be positive, got {config['lr']}")
    return config


def run_directory(config):
    return Path(config["save_dir"]) / config["scale_label"] / f"seed_{config['seeds']['run_seed']}" / config["model"]


def make_datasets(config):
    """Materialize every split for one run; identical for both models given the same seeds."""
    gen = NearestNeighborConfig(**{**config["generation"], "best_similarity_range":
                                   tuple(config["generation"]["best_similarity_range"])})
    seeds, length = config["seeds"], config["sequence_length"]
    pool = make_train_pool(config["train_size"], length, seeds["data"], gen)
    train, val, train_idx, val_idx = split_train_val(pool, config["val_size"], seeds["split"])
    test = make_balanced_suite(config["test_size"], length, seeds["data"], "test", gen)
    datasets = {"train": train, "val": val, "test": test}
    if config["run_extrapolation"]:
        datasets["extrapolation"] = make_balanced_suite(
            config["extrapolation_size"], config["extrapolation_length"], seeds["data"], "extrapolation", gen)
    return datasets


def dataset_manifest(config, datasets):
    return {
        "generation": config["generation"],
        "sequence_length": config["sequence_length"],
        "n_candidates": config["n_candidates"],
        "seeds": config["seeds"],
        "splits": {
            name: {
                "n_sequences": len(ds),
                "sequence_length": ds.sequence_length,
                "case_counts": ds.case_counts(),
                "checksums": ds.checksums(),
            }
            for name, ds in datasets.items()
        },
        "checksum_method": "sha256 over dtype, shape, and little-endian tensor bytes",
    }


# ---------------------------------------------------------------------------------------------
# Loss and finite checks
# ---------------------------------------------------------------------------------------------

def masked_mse(outputs, targets, mask):
    """Unweighted mean of squared errors over the True entries of `mask`."""
    if not (outputs.shape == targets.shape == mask.shape):
        raise ValueError(
            f"shape mismatch: outputs {tuple(outputs.shape)}, targets {tuple(targets.shape)}, "
            f"mask {tuple(mask.shape)}"
        )
    return ((outputs - targets) ** 2)[mask].mean()


def _require_finite(tensor, what, context):
    if not torch.isfinite(tensor).all():
        raise NonFiniteError(f"non-finite {what} at {context}")


def _context(config, epoch, batch):
    return f"run_seed={config['seeds']['run_seed']} model={config['model']} epoch={epoch} batch={batch}"


def _model_forward(model, batch, device):
    """The only model input is `inputs`; everything else in the batch is analysis metadata."""
    return model(batch["inputs"].to(device))[0]


def _digest_ids(sequence_ids):
    return hashlib.sha256(sequence_ids.numpy().astype("<i8").tobytes()).hexdigest()


# ---------------------------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------------------------

@torch.no_grad()
def validation_metrics(model, dataset, config, epoch):
    """MSE over metric_mask (selection metric) and over loss_mask (training-loss basis)."""
    model.eval()
    device, sums = config["device"], np.zeros(4)
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)
    for b, batch in enumerate(loader):
        outputs = _model_forward(model, batch, device)
        _require_finite(outputs, "validation outputs", _context(config, epoch, f"val{b}"))
        sq = ((outputs - batch["targets"].to(device)) ** 2)
        for k, key in enumerate(("metric_mask", "loss_mask")):
            mask = batch[key].to(device)
            sums[2 * k] += sq[mask].sum().item()
            sums[2 * k + 1] += mask.sum().item()
    return {"val_mse": float(sums[0] / sums[1]), "val_loss_mask_mse": float(sums[2] / sums[3])}


@torch.no_grad()
def predict_dataset(model, dataset, config, batch_size=None):
    """Predictions and hidden-mean write proportions for every timestep: two [N, L] arrays."""
    model.eval()
    device = config["device"]
    loader = DataLoader(dataset, batch_size=batch_size or config["batch_size"], shuffle=False)
    preds, alphas = [], []
    for b, batch in enumerate(loader):
        outputs, _, diag = model(batch["inputs"].to(device), return_diagnostics=True)
        alpha = diag["alpha"]  # [batch, L, hidden], computed per hidden unit
        context = f"prediction batch {b}"
        _require_finite(outputs, "prediction outputs", context)
        _require_finite(alpha, "alpha", context)
        if (alpha < -ALPHA_TOLERANCE).any() or (alpha > 1 + ALPHA_TOLERANCE).any():
            raise ValueError(f"alpha outside [0, 1] at {context}: [{alpha.min().item()}, {alpha.max().item()}]")
        preds.append(outputs[..., 0].cpu())
        alphas.append(alpha.mean(dim=-1).cpu())  # reduce only after the elementwise ratio
    return torch.cat(preds).numpy().astype(np.float64), torch.cat(alphas).numpy().astype(np.float64)


def prediction_frame(dataset, pred, alpha, run_seed, model_name, split):
    """Tidy per-timestep table. No raw input/forget gate values are included."""
    n, length = pred.shape
    t = dataset.tensors
    sims = t["similarities"][..., 0].numpy().astype(np.float64)
    targets = t["targets"][..., 0].numpy().astype(np.float64)
    is_metric = t["metric_mask"][..., 0].numpy()
    is_event = t["event_mask"][..., 0].numpy()
    best_idx = t["best_indices"].numpy()

    candidate = np.zeros((n, length), dtype=bool)
    candidate[:, 1:] = True
    running_best = np.full((n, length), np.nan)
    running_best[:, 1:] = np.maximum.accumulate(sims[:, 1:], axis=1)
    sims_out = np.where(candidate, sims, np.nan)
    targets_out = np.where(candidate, targets, np.nan)
    err = pred - targets_out

    timestep = np.tile(np.arange(length), n)
    return pd.DataFrame({
        "run_seed": run_seed,
        "model": model_name,
        "split": split,
        "case_type": np.repeat(dataset.case_types, length),
        "sequence_id": np.repeat(dataset.sequence_ids.numpy(), length),
        "sequence_length": length,
        "timestep": timestep,
        "candidate_index": timestep,  # candidate t = token t; 0 marks the reference token
        "similarity": sims_out.ravel(),
        "running_best_similarity": running_best.ravel(),
        "is_event": is_event.ravel(),
        "is_metric_step": is_metric.ravel(),
        "best_index": best_idx.ravel(),
        "target": targets_out.ravel(),
        "prediction": pred.ravel(),
        "absolute_error": np.abs(err).ravel(),
        "squared_error": (err**2).ravel(),
        "alpha_mean": alpha.ravel(),
    })


def evaluate_split(model, dataset, split, config, run_dir):
    """Run one held-out evaluation pass and write its prediction and metric artifacts."""
    run_seed, name = config["seeds"]["run_seed"], config["model"]
    pred, alpha = predict_dataset(model, dataset, config)
    frame = prediction_frame(dataset, pred, alpha, run_seed, name, split)
    frame.to_csv(run_dir / f"predictions_{split}.csv", index=False)

    t = dataset.tensors
    per_seq = nnm.sequence_metrics(
        pred, t["targets"][..., 0].numpy(), t["event_mask"][..., 0].numpy(),
        t["metric_mask"][..., 0].numpy(), alpha)
    per_seq = nnm.add_identity(per_seq, run_seed, name, split, dataset.case_types, dataset.sequence_ids.numpy())
    per_seed = nnm.aggregate_per_seed(per_seq)
    per_seq.to_csv(run_dir / f"per_sequence_metrics_{split}.csv", index=False)
    per_seed.to_csv(run_dir / f"per_seed_metrics_{split}.csv", index=False)
    return per_seed


# ---------------------------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------------------------

def train_model(model, train_ds, val_ds, config, run_dir):
    """Adam training with masked MSE for the full epoch budget; keeps the best-validation checkpoint."""
    device = config["device"]
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=config["lr"], weight_decay=config["weight_decay"])
    loader_gen = torch.Generator().manual_seed(config["seeds"]["loader"])
    loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, generator=loader_gen)
    clip = config["grad_clip"]

    history, best_val, best_epoch = [], math.inf, 0
    order_hash = hashlib.sha256()
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        start = time.perf_counter()
        losses, norms, clipped, n_seen = [], [], 0, 0
        for b, batch in enumerate(loader):
            context = _context(config, epoch, b)
            if epoch == 1:
                order_hash.update(batch["sequence_id"].numpy().astype("<i8").tobytes())
            inputs = batch["inputs"].to(device)
            targets, loss_mask = batch["targets"].to(device), batch["loss_mask"].to(device)
            _require_finite(inputs, "inputs", context)

            optimizer.zero_grad()
            outputs = model(inputs)[0]
            _require_finite(outputs, "outputs", context)
            loss = masked_mse(outputs, targets, loss_mask)
            _require_finite(loss, "loss", context)
            loss.backward()
            for name, p in model.named_parameters():
                if p.grad is not None:
                    _require_finite(p.grad, f"gradient of {name}", context)
            norm = float(torch.nn.utils.clip_grad_norm_(params, max_norm=clip if clip > 0 else float("inf")))
            _require_finite(torch.tensor(norm), "gradient norm", context)
            clipped += int(clip > 0 and norm > clip)
            optimizer.step()

            losses.append(loss.item() * inputs.shape[0])
            norms.append(norm)
            n_seen += inputs.shape[0]

        val = validation_metrics(model, val_ds, config, epoch)
        row = {
            "epoch": epoch,
            "train_loss": sum(losses) / n_seen,
            **val,
            "grad_norm_mean": float(np.mean(norms)),
            "grad_norm_max": float(np.max(norms)),
            "clipped_batches": clipped,
            "n_batches": len(norms),
            "clip_fraction": clipped / len(norms),
            "epoch_seconds": time.perf_counter() - start,
        }
        improved = val["val_mse"] < best_val
        row["is_best"] = bool(improved)
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        print(f"[{config['model']} seed {config['seeds']['run_seed']}] epoch {epoch:03d} "
              f"train {row['train_loss']:.5f} val {val['val_mse']:.5f} "
              f"gnorm {row['grad_norm_mean']:.3g} ({row['epoch_seconds']:.1f}s)" + (" *" if improved else ""),
              flush=True)

        payload = {"epoch": epoch, "model_state_dict": copy.deepcopy(model.state_dict()),
                   "val_mse": val["val_mse"], "config": config}
        if improved:
            best_val, best_epoch = val["val_mse"], epoch
            torch.save(payload, ckpt_dir / "best.pt")
        torch.save({**payload, "optimizer_state_dict": optimizer.state_dict()}, ckpt_dir / "last.pt")

    return {
        "history": history, "best_epoch": best_epoch, "best_val_mse": best_val,
        "epochs_run": len(history), "loader_order_checksum_epoch1": order_hash.hexdigest(),
    }


# ---------------------------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------------------------

def parameter_checksums(model):
    return {name: hashlib.sha256(p.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for name, p in model.state_dict().items()}


def source_revision():
    """Git commit when available, otherwise a hash of the source tree (this checkout is not a repo)."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                         stderr=subprocess.DEVNULL).decode().strip()
        return {"kind": "git", "commit": commit}
    except Exception:
        h = hashlib.sha256()
        files = sorted(list((REPO_ROOT / "src").rglob("*.py")) + list((REPO_ROOT / "scripts").rglob("*.py")))
        for path in files:
            h.update(str(path.relative_to(REPO_ROOT)).encode())
            h.update(path.read_bytes())
        return {"kind": "source-tree-sha256", "sha256": h.hexdigest(), "n_files": len(files)}


def environment_info():
    try:
        import pennylane
        pennylane_version = pennylane.__version__
    except Exception:
        pennylane_version = "unavailable"
    return {
        "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
        "pennylane": pennylane_version, "numpy": np.__version__, "pandas": pd.__version__,
        "torch_threads": torch.get_num_threads(), "cuda_available": torch.cuda.is_available(),
    }


def _write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


# ---------------------------------------------------------------------------------------------
# One full run
# ---------------------------------------------------------------------------------------------

def run_experiment(config, run_dir=None):
    """Train one model on one seed, evaluate the best-validation checkpoint once, write artifacts."""
    run_dir = Path(run_dir) if run_dir is not None else run_directory(config)
    previous_config = run_dir / "config.json"
    if previous_config.exists():
        previous = json.loads(previous_config.read_text(encoding="utf-8"))
        if previous.get("qslstm_recurrence") != config.get("qslstm_recurrence"):
            raise ValueError("Existing run uses a different Q-sLSTM recurrence; choose a new --save-dir")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "complete.json").unlink(missing_ok=True)
    _write_json(run_dir / "config.json", config)
    _write_json(run_dir / "environment.json", environment_info())
    _write_json(run_dir / "source_revision.json", source_revision())
    started = time.perf_counter()
    timing = {}
    try:
        t0 = time.perf_counter()
        datasets = make_datasets(config)
        manifest = dataset_manifest(config, datasets)
        _write_json(run_dir / "dataset_manifest.json", manifest)
        timing["data_seconds"] = time.perf_counter() - t0

        model = build_quantum_model(
            config["model"], config["input_size"], config["hidden_size"], config["output_size"],
            config["qnn_depth"], gate_epsilon=config["gate_epsilon"], device=config["device"],
            seed=config["seeds"]["model"])
        n_params = count_trainable_parameters(model)
        _write_json(run_dir / "parameters.json", {
            "trainable_parameters": n_params, "n_qubits": config["n_qubits"],
            "hidden_size": config["hidden_size"], "qnn_depth": config["qnn_depth"],
            "parameter_shapes": {k: list(v.shape) for k, v in model.state_dict().items()},
        })
        _write_json(run_dir / "init_parameters.json", {
            "model_seed": config["seeds"]["model"],
            "checksums": parameter_checksums(model),
            "note": "sha256 of each parameter tensor before training; paired models must match name-for-name",
        })

        t0 = time.perf_counter()
        result = train_model(model, datasets["train"], datasets["val"], config, run_dir)
        timing["train_seconds"] = time.perf_counter() - t0
        _write_json(run_dir / "train_summary.json",
                    {k: v for k, v in result.items() if k != "history"})

        # Held-out data is touched only now, once, with the checkpoint chosen on validation.
        best = torch.load(run_dir / "checkpoints" / "best.pt", map_location=config["device"])
        model.load_state_dict(best["model_state_dict"])
        t0 = time.perf_counter()
        per_seed = {"test": evaluate_split(model, datasets["test"], "test", config, run_dir)}
        if "extrapolation" in datasets:
            per_seed["extrapolation"] = evaluate_split(
                model, datasets["extrapolation"], "extrapolation", config, run_dir)
        timing["evaluation_seconds"] = time.perf_counter() - t0
        timing["total_seconds"] = time.perf_counter() - started
        _write_json(run_dir / "timing.json", timing)

        headline = per_seed["test"].query("case_type == 'all'").iloc[0]
        _write_json(run_dir / "complete.json", {
            "model": config["model"], "run_seed": config["seeds"]["run_seed"],
            "scale_label": config["scale_label"], "best_epoch": result["best_epoch"],
            "best_val_mse": result["best_val_mse"], "test_mse": float(headline["mse"]),
            "trainable_parameters": n_params,
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
    """Compare two runs of the same seed: data, splits, loader order, and initial parameters."""
    load = lambda d, f: json.loads((Path(d) / f).read_text(encoding="utf-8"))
    ma, mb = load(run_dir_a, "dataset_manifest.json"), load(run_dir_b, "dataset_manifest.json")
    ia, ib = load(run_dir_a, "init_parameters.json"), load(run_dir_b, "init_parameters.json")
    sa, sb = load(run_dir_a, "train_summary.json"), load(run_dir_b, "train_summary.json")
    ca, cb = load(run_dir_a, "config.json"), load(run_dir_b, "config.json")
    shared = sorted(set(ia["checksums"]) & set(ib["checksums"]))
    return {
        "recurrence_version_equal": ca.get("qslstm_recurrence") == cb.get("qslstm_recurrence"),
        "dataset_checksums_equal": ma["splits"] == mb["splits"],
        "seed_map_equal": ma["seeds"] == mb["seeds"],
        "loader_order_equal": sa["loader_order_checksum_epoch1"] == sb["loader_order_checksum_epoch1"],
        "shared_parameter_names": shared,
        "initial_shared_parameters_equal": bool(shared) and all(
            ia["checksums"][k] == ib["checksums"][k] for k in shared),
        "all_parameter_names_shared": set(ia["checksums"]) == set(ib["checksums"]),
    }
