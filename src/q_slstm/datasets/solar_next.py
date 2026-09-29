# Real-data 1-D input / 1-D output task for the scalar-task pipeline.
#
#     solar_next   y[t] = g[t + 1] from inputs g[1..t]: next-hour Ontario (IESO) solar generation
#
# The series is the hourly IESO solar output already collected for the solar-generation experiment
# (data/solar_generation, 2024-01-01 .. 2025-12-31 UTC). Each sequence is a window of L consecutive
# hours; the input is that hour's generation, the target the next hour's. Both are scaled to [-1, 1]
# with the training period's maximum (g_scaled = 2 g / max_train - 1).
#
# Splits are chronological and match the solar-generation experiment's split_boundaries (70 / 10 / 20
# % of the hourly grid); a window belongs to a split when all its L + 1 rows (inputs and the last
# target) lie inside it, so no window straddles a boundary. Windows touching a flagged or missing row
# are excluded. Training windows are sampled per data seed; validation, test, and extrapolation
# windows are evenly spaced over their period and identical for every seed and model.

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .scalar_tasks import STREAM_ID_OFFSETS, ScalarTaskDataset

REAL_TASKS = ("solar_next",)
DEFAULT_SOLAR_ROOT = Path(__file__).resolve().parents[3] / "data" / "solar_generation"
DAY = 24


def _series(root, scale="paper"):
    root = Path(root)
    pointer = json.loads((root / f"prepared_{scale}.json").read_text(encoding="utf-8"))
    processed = root / pointer["processed_dir"].replace("\\", "/")
    manifest = json.loads((processed / "dataset_manifest.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(processed / "normalized" / "ieso_solar_generation.csv")
    return frame, manifest["split_boundaries"], pointer["dataset_id"]


def _eligible_starts(ok, lo, hi, length):
    """Window starts s with rows s..s+length (inputs + last target) all valid and inside [lo, hi)."""
    starts = np.arange(lo, hi - length)
    bad = np.concatenate([[0], np.cumsum(~ok)])
    return starts[bad[starts + length + 1] - bad[starts] == 0]


def _even(starts, count):
    if count >= len(starts):
        return starts
    return starts[np.round(np.linspace(0, len(starts) - 1, count)).astype(int)]


def _windows(series, starts, length, split, stream):
    idx = starts[:, None] + np.arange(length + 1)[None, :]
    window = series[idx]
    tensors = {
        "inputs": torch.from_numpy(window[:, :-1].astype(np.float32))[..., None],
        "targets": torch.from_numpy(window[:, 1:].astype(np.float32))[..., None],
        "loss_mask": torch.ones(len(starts), length, 1, dtype=torch.bool),
    }
    metadata = {"task": "solar_next", "stream": stream, "split": split, "sequence_length": length,
                "warmup_steps": 0, "window_start_rows": starts.tolist()}
    return ScalarTaskDataset(tensors, STREAM_ID_OFFSETS[stream] + starts, "solar_next", metadata)


def reference_mse(dataset):
    """Classical next-hour references on a dataset's supervised steps (scaled units).

    persistence       g[t+1] ~ g[t]
    daily_persistence g[t+1] ~ g[t+1-24], scored on steps t >= 23 where that hour is in the window
    """
    x, y = dataset.tensors["inputs"][..., 0].double(), dataset.tensors["targets"][..., 0].double()
    out = {"persistence": float(((x - y) ** 2).mean())}
    if x.shape[1] > DAY - 1:
        out["daily_persistence"] = float(((x[:, : x.shape[1] - DAY + 1] - y[:, DAY - 1:]) ** 2).mean())
        out["persistence_same_steps_as_daily"] = float(((x[:, DAY - 1:] - y[:, DAY - 1:]) ** 2).mean())
    return out


def load_solar_next(config, data_seed, root=DEFAULT_SOLAR_ROOT):
    """train / val / test (/ extrapolation) ScalarTaskDatasets for one scalar-task run config."""
    frame, bounds, dataset_id = _series(root)
    ok = (frame["status"] == "ok").to_numpy() & frame["solar_generation"].notna().to_numpy()
    raw = frame["solar_generation"].to_numpy(dtype=np.float64)
    t_lo, t_hi = bounds["train"]["rows"]
    scale = float(raw[t_lo:t_hi][ok[t_lo:t_hi]].max())
    series = 2.0 * raw / scale - 1.0

    length = config["sequence_length"]
    rng = np.random.default_rng(np.random.SeedSequence([int(data_seed), 314]))
    train_starts = _eligible_starts(ok, t_lo, t_hi, length)
    n_train = config["optimizer_train_size"]
    if n_train > len(train_starts):
        raise ValueError(f"solar_next has {len(train_starts)} training windows, {n_train} requested")
    datasets = {"train": _windows(series, np.sort(rng.choice(train_starts, n_train, replace=False)),
                                  length, "train", "train_pool")}
    for split, stream, size in (("val", "train_pool", config["val_size"]), ("test", "test", config["test_size"])):
        lo, hi = bounds[split]["rows"]
        starts = _even(_eligible_starts(ok, lo, hi, length), size)
        datasets[split] = _windows(series, starts, length, split, stream)
    if config.get("run_extrapolation"):
        lo, hi = bounds["test"]["rows"]
        ext = config["extrapolation_length"]
        datasets["extrapolation"] = _windows(series, _even(_eligible_starts(ok, lo, hi, ext),
                                                           config["extrapolation_size"]),
                                             ext, "extrapolation", "extrapolation")
    source = {"dataset_id": dataset_id, "scale_mw": scale, "scaling": "2 * MW / scale_mw - 1",
              "split_boundaries": bounds, "excluded": "windows touching a row with status != ok"}
    for ds in datasets.values():
        ds.metadata["source"] = source
    return datasets
