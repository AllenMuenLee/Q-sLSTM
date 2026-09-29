# Training / evaluation pipeline for the 1-D scalar sequence tasks.
#
# Compares the conventional QLSTM, the Q-sLSTM variants, and (as a sanity reference) a classical
# nn.LSTM on the tasks in q_slstm.datasets.scalar_tasks. Every model maps a [batch, L, 1] input to a
# [batch, L, 1] prediction; training uses masked MSE over the task's supervised steps.
#
# Run layout: <save_dir>/<scale>/<run date>/<task>/seed_<seed>/<model>/

from __future__ import annotations

import copy
import datetime
import hashlib
import math
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from q_slstm.datasets.scalar_tasks import (
    INPUT_SIZE,
    OUTPUT_SIZE,
    TASKS,
    ScalarTaskConfig,
    generate_dataset,
    split_train_val,
)
from q_slstm.experiments import scalar_tasks_metrics as stm
from q_slstm.experiments.nearest_neighbor import (
    _require_finite,
    _write_json,
    derive_seeds,
    environment_info,
    masked_mse,
    parameter_checksums,
    source_revision,
)
from q_slstm.models.factory import QUANTUM_MODELS, build_quantum_model, count_trainable_parameters
from q_slstm.models.q_slstm_cell import DEFAULT_GATE_EPSILON, QSLSTM_RECURRENCE
from q_slstm.models.q_slstm_log_cell import QSLSTM_LOG_RECURRENCE

CLASSICAL_MODELS = ("lstm",)
MODELS = QUANTUM_MODELS + CLASSICAL_MODELS

# The paper preset keeps the nearest-neighbor model size (hidden 6, depth 3) and length (32) so the
# two experiments are comparable; tasks are simpler, so the data and epoch budgets are smaller.
PRESETS = {
    "paper": dict(
        sequence_length=32, train_size=1600, val_fraction=0.125, test_size=400,
        extrapolation_length=64, extrapolation_size=200, hidden_size=4, qnn_depth=2,
        batch_size=64, epochs=60, lr=1e-2, n_seeds=20,
    ),
    "pilot": dict(
        sequence_length=12, train_size=32, val_fraction=0.125, test_size=8,
        extrapolation_length=24, extrapolation_size=4, hidden_size=2, qnn_depth=1,
        batch_size=8, epochs=2, lr=1e-2, n_seeds=2,
    ),
}
PRESET_FIELDS = tuple(k for k in PRESETS["paper"] if k != "n_seeds")


# ---------------------------------------------------------------------------------------------
# Arguments and config
# ---------------------------------------------------------------------------------------------

def add_run_arguments(parser):
    """Arguments shared by the training script and the sweep launcher (all optional here)."""
    a = parser.add_argument
    d = ScalarTaskConfig()
    a("--scale", choices=sorted(PRESETS), default="paper", help="scale preset (default: paper)")
    a("--sequence-length", type=int, default=None, help="tokens per sequence (paper: 32)")
    a("--train-size", type=int, default=None, help="pool size = training + validation (paper: 1600)")
    a("--val-fraction", type=float, default=None, help="fraction of the pool used for validation")
    a("--test-size", type=int, default=None, help="held-out test size (paper: 400)")
    a("--extrapolation-length", type=int, default=None, help="tokens per extrapolation sequence (paper: 64)")
    a("--extrapolation-size", type=int, default=None, help="extrapolation suite size (paper: 200)")
    a("--run-extrapolation", action="store_true", help="also evaluate the checkpoint at the extrapolation length")
    a("--data-seed", type=int, default=None, help="base seed for data and split (default: derived from --seed)")
    a("--hidden-size", type=int, default=None)
    a("--qnn-depth", type=int, default=None, help="VQC depth (ignored by the classical lstm)")
    a("--gate-epsilon", type=float, default=DEFAULT_GATE_EPSILON)
    a("--batch-size", type=int, default=None)
    a("--epochs", type=int, default=None)
    a("--lr", type=float, default=None, help="overrides the preset learning rate")
    a("--weight-decay", type=float, default=0.0)
    a("--grad-clip", type=float, default=0.0, help="max gradient norm; 0 disables clipping (norms are still logged)")
    a("--delay", type=int, default=d.delay, help="delay task: lag k")
    a("--ema-decay", type=float, default=d.ema_decay, help="ema task: decay d")
    a("--flip-probability", type=float, default=d.flip_probability, help="flip_flop task: pulse probability")
    a("--narma-order", type=int, default=d.narma_order, help="narma task: order n")
    a("--device", choices=["cpu", "cuda"], default="cpu")
    a("--save-dir", type=str, default="results/scalar_tasks")
    a("--run-date", type=str, default=None, help="date folder (YYYY-MM-DD) under <save-dir>/<scale>; default: today")


def generation_config(config):
    gen = dict(config["generation"])
    gen["sine_frequency_range"] = tuple(gen["sine_frequency_range"])
    return ScalarTaskConfig(**gen)


def resolve_config(args):
    """Merge the preset with explicit overrides into one fully resolved, validated config dict."""
    a = dict(vars(args)) if not isinstance(args, dict) else dict(args)
    if a.get("model") not in MODELS:
        raise ValueError(f"--model must be one of {MODELS}, got {a.get('model')!r}")
    if a.get("task") not in TASKS:
        raise ValueError(f"--task must be one of {TASKS}, got {a.get('task')!r}")

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

    d = ScalarTaskConfig()
    generation = ScalarTaskConfig(
        delay=a.get("delay", d.delay), ema_decay=a.get("ema_decay", d.ema_decay),
        flip_probability=a.get("flip_probability", d.flip_probability),
        narma_order=a.get("narma_order", d.narma_order),
    )
    generation.validate(a["task"], resolved["sequence_length"])
    if a.get("run_extrapolation"):
        generation.validate(a["task"], resolved["extrapolation_length"])

    val_size = int(round(resolved["train_size"] * resolved["val_fraction"]))
    if not 0 < val_size < resolved["train_size"]:
        raise ValueError("train_size and val_fraction leave no training or validation sequences")
    run_date = a.get("run_date") or datetime.date.today().isoformat()
    try:
        datetime.date.fromisoformat(run_date)
    except ValueError:
        raise ValueError(f"--run-date must be YYYY-MM-DD, got {run_date!r}") from None
    model = a["model"]
    config = {
        "kind": "scalar_task_run",
        "task": a["task"],
        "model": model,
        "scale_preset": preset,
        "scale_label": preset if not overrides else f"{preset}-overridden",
        "preset_overrides": overrides,
        **resolved,
        "val_size": val_size,
        "optimizer_train_size": resolved["train_size"] - val_size,
        "early_stopping": "none: every run trains for the full epoch budget; best-validation checkpoint kept",
        "run_extrapolation": bool(a.get("run_extrapolation", False)),
        "input_size": INPUT_SIZE,
        "output_size": OUTPUT_SIZE,
        "n_qubits": INPUT_SIZE + resolved["hidden_size"] if model in QUANTUM_MODELS else None,
        "gate_epsilon": a.get("gate_epsilon", DEFAULT_GATE_EPSILON),
        "qslstm_recurrence": (QSLSTM_LOG_RECURRENCE if model == "qslstm_log"
                              else QSLSTM_RECURRENCE if model == "qslstm" else None),
        "weight_decay": a.get("weight_decay", 0.0),
        "grad_clip": a.get("grad_clip", 0.0),
        "device": a.get("device", "cpu"),
        "generation": generation.to_dict(),
        "seeds": derive_seeds(int(a.get("seed", 0)), a.get("data_seed")),
        "save_dir": str(a.get("save_dir", "results/scalar_tasks")),
        "run_date": run_date,
        "loss": "mean squared error over loss_mask (warm-up steps excluded)",
        "selection_metric": "validation MSE over loss_mask",
    }
    for key in ("epochs", "batch_size"):
        if config[key] < 1:
            raise ValueError(f"{key} must be positive, got {config[key]}")
    if not config["lr"] > 0:
        raise ValueError(f"lr must be positive, got {config['lr']}")
    return config


def sweep_directory(config):
    """<save_dir>/<scale>/<run date>: one dated folder per sweep, so later sweeps never overwrite it."""
    return Path(config["save_dir"]) / config["scale_label"] / config["run_date"]


def run_directory(config):
    return sweep_directory(config) / config["task"] / f"seed_{config['seeds']['run_seed']}" / config["model"]


def make_datasets(config):
    """Materialize every split for one run; identical for all models given the same seeds and task."""
    gen, seeds, task = generation_config(config), config["seeds"], config["task"]
    pool = generate_dataset(task, config["train_size"], config["sequence_length"], seeds["data"], "train_pool", gen)
    train, val, _, _ = split_train_val(pool, config["val_size"], seeds["split"])
    datasets = {
        "train": train, "val": val,
        "test": generate_dataset(task, config["test_size"], config["sequence_length"], seeds["data"], "test", gen),
    }
    if config["run_extrapolation"]:
        datasets["extrapolation"] = generate_dataset(
            task, config["extrapolation_size"], config["extrapolation_length"], seeds["data"], "extrapolation", gen)
    return datasets


def dataset_manifest(config, datasets):
    return {
        "task": config["task"],
        "generation": config["generation"],
        "seeds": config["seeds"],
        "splits": {name: {"n_sequences": len(ds), "sequence_length": ds.sequence_length,
                          "warmup_steps": ds.metadata["warmup_steps"], "checksums": ds.checksums()}
                   for name, ds in datasets.items()},
        "checksum_method": "sha256 over dtype, shape, and little-endian tensor bytes",
    }


def training_target_mean(dataset):
    t = dataset.tensors
    return float(t["targets"][t["loss_mask"]].double().mean())


# ---------------------------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------------------------

class ClassicalLSTM(nn.Module):
    """nn.LSTM + linear read-out with the quantum models' (outputs, state) return contract."""

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h, state = self.lstm(x)
        return self.head(h), state


def build_model(config):
    if config["model"] in QUANTUM_MODELS:
        return build_quantum_model(
            config["model"], config["input_size"], config["hidden_size"], config["output_size"],
            config["qnn_depth"], gate_epsilon=config["gate_epsilon"], device=config["device"],
            seed=config["seeds"]["model"])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config["seeds"]["model"])
        net = ClassicalLSTM(config["input_size"], config["hidden_size"], config["output_size"])
    return net.to(config["device"])


def _context(config, epoch, batch):
    return (f"task={config['task']} run_seed={config['seeds']['run_seed']} model={config['model']} "
            f"epoch={epoch} batch={batch}")


# ---------------------------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------------------------

@torch.no_grad()
def validation_mse(model, dataset, config, epoch):
    model.eval()
    device, total, count = config["device"], 0.0, 0
    for b, batch in enumerate(DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)):
        outputs = model(batch["inputs"].to(device))[0]
        _require_finite(outputs, "validation outputs", _context(config, epoch, f"val{b}"))
        mask = batch["loss_mask"].to(device)
        total += ((outputs - batch["targets"].to(device)) ** 2)[mask].sum().item()
        count += mask.sum().item()
    return total / count


@torch.no_grad()
def predict_dataset(model, dataset, config):
    """Predictions for every timestep: [N, L] float64 array."""
    model.eval()
    preds = []
    for b, batch in enumerate(DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)):
        outputs = model(batch["inputs"].to(config["device"]))[0]
        _require_finite(outputs, "prediction outputs", f"prediction batch {b}")
        preds.append(outputs[..., 0].cpu())
    return torch.cat(preds).numpy().astype(np.float64)


def evaluate_split(model, dataset, split, config, run_dir, baseline_value):
    """One held-out pass: writes predictions, per-sequence, per-seed, and per-timestep tables."""
    run_seed, name, task = config["seeds"]["run_seed"], config["model"], config["task"]
    pred = predict_dataset(model, dataset, config)
    t = dataset.tensors
    inputs = t["inputs"][..., 0].numpy().astype(np.float64)
    target = t["targets"][..., 0].numpy().astype(np.float64)
    mask = t["loss_mask"][..., 0].numpy()
    n, length = pred.shape

    pd.DataFrame({
        "run_seed": run_seed, "model": name, "task": task, "split": split,
        "sequence_id": np.repeat(dataset.sequence_ids.numpy(), length),
        "timestep": np.tile(np.arange(length), n),
        "input": inputs.ravel(), "target": target.ravel(), "prediction": pred.ravel(),
        "is_supervised": mask.ravel(),
    }).to_csv(run_dir / f"predictions_{split}.csv", index=False)

    per_seq = stm.add_identity(stm.sequence_metrics(pred, target, mask), run_seed, name, task, split,
                               dataset.sequence_ids.numpy())
    per_seed = stm.aggregate_per_seed(per_seq, target, mask, baseline_value)
    per_seq.to_csv(run_dir / f"per_sequence_metrics_{split}.csv", index=False)
    per_seed.to_csv(run_dir / f"per_seed_metrics_{split}.csv", index=False)
    pd.DataFrame({"timestep": np.arange(length), "mse": stm.timestep_curve(pred, target, mask)}).to_csv(
        run_dir / f"timestep_mse_{split}.csv", index=False)
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
            clipped += int(clip > 0 and norm > clip)
            optimizer.step()

            losses.append(loss.item() * inputs.shape[0])
            norms.append(norm)
            n_seen += inputs.shape[0]

        val = validation_mse(model, val_ds, config, epoch)
        improved = val < best_val
        history.append({
            "epoch": epoch, "train_loss": sum(losses) / n_seen, "val_mse": val,
            "grad_norm_mean": float(np.mean(norms)), "grad_norm_max": float(np.max(norms)),
            "clip_fraction": clipped / len(norms), "epoch_seconds": time.perf_counter() - start,
            "is_best": bool(improved),
        })
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        row = history[-1]
        print(f"[{config['task']} {config['model']} seed {config['seeds']['run_seed']}] epoch {epoch:03d} "
              f"train {row['train_loss']:.5f} val {val:.5f} gnorm {row['grad_norm_mean']:.3g} "
              f"({row['epoch_seconds']:.1f}s)" + (" *" if improved else ""), flush=True)

        payload = {"epoch": epoch, "model_state_dict": copy.deepcopy(model.state_dict()),
                   "val_mse": val, "config": config}
        if improved:
            best_val, best_epoch = val, epoch
            torch.save(payload, ckpt_dir / "best.pt")
        torch.save({**payload, "optimizer_state_dict": optimizer.state_dict()}, ckpt_dir / "last.pt")

    return {"history": history, "best_epoch": best_epoch, "best_val_mse": best_val,
            "epochs_run": len(history), "loader_order_checksum_epoch1": order_hash.hexdigest()}


# ---------------------------------------------------------------------------------------------
# One full run
# ---------------------------------------------------------------------------------------------

def run_experiment(config, run_dir=None):
    """Train one model on one task and seed, evaluate the best-validation checkpoint once, write artifacts."""
    run_dir = Path(run_dir) if run_dir is not None else run_directory(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "complete.json").unlink(missing_ok=True)
    _write_json(run_dir / "config.json", config)
    _write_json(run_dir / "environment.json", environment_info())
    _write_json(run_dir / "source_revision.json", source_revision())
    started, timing = time.perf_counter(), {}
    try:
        t0 = time.perf_counter()
        datasets = make_datasets(config)
        _write_json(run_dir / "dataset_manifest.json", dataset_manifest(config, datasets))
        baseline_value = training_target_mean(datasets["train"])
        timing["data_seconds"] = time.perf_counter() - t0

        model = build_model(config)
        n_params = count_trainable_parameters(model)
        _write_json(run_dir / "parameters.json", {
            "trainable_parameters": n_params, "n_qubits": config["n_qubits"],
            "hidden_size": config["hidden_size"], "qnn_depth": config["qnn_depth"],
            "parameter_shapes": {k: list(v.shape) for k, v in model.state_dict().items()},
        })
        _write_json(run_dir / "init_parameters.json", {
            "model_seed": config["seeds"]["model"], "checksums": parameter_checksums(model)})

        t0 = time.perf_counter()
        result = train_model(model, datasets["train"], datasets["val"], config, run_dir)
        timing["train_seconds"] = time.perf_counter() - t0
        _write_json(run_dir / "train_summary.json",
                    {**{k: v for k, v in result.items() if k != "history"},
                     "baseline_constant_prediction": baseline_value})

        # Held-out data is touched only now, once, with the checkpoint chosen on validation.
        best = torch.load(run_dir / "checkpoints" / "best.pt", map_location=config["device"])
        model.load_state_dict(best["model_state_dict"])
        t0 = time.perf_counter()
        per_seed = {split: evaluate_split(model, datasets[split], split, config, run_dir, baseline_value)
                    for split in ("test", "extrapolation") if split in datasets}
        timing["evaluation_seconds"] = time.perf_counter() - t0
        timing["total_seconds"] = time.perf_counter() - started
        _write_json(run_dir / "timing.json", timing)

        headline = per_seed["test"].iloc[0]
        _write_json(run_dir / "complete.json", {
            "task": config["task"], "model": config["model"], "run_seed": config["seeds"]["run_seed"],
            "scale_label": config["scale_label"], "best_epoch": result["best_epoch"],
            "best_val_mse": result["best_val_mse"], "test_mse": float(headline["mse"]),
            "test_nmse": float(headline["nmse"]), "test_skill": float(headline["skill"]),
            "trainable_parameters": n_params,
        })
        return run_dir
    except BaseException as exc:
        _write_json(run_dir / "failure.json", {
            "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(),
            "task": config["task"], "model": config["model"], "run_seed": config["seeds"]["run_seed"],
            "elapsed_seconds": time.perf_counter() - started,
        })
        raise
