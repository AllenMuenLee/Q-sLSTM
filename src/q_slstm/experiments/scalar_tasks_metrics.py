# Metrics for the 1-D scalar sequence tasks.
#
# Every metric uses only rows where `loss_mask` is true (warm-up steps never contribute). Sequences
# are summarized first; per-seed tables then average sequences (never timesteps). Because the tasks
# have very different target scales, every table also reports errors relative to two references:
#     nmse   = mse / Var(target)            (1.0 = no better than predicting the test-set mean)
#     skill  = 1 - mse / baseline_mse       (baseline = constant prediction of the training-target mean)

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from q_slstm.experiments.nearest_neighbor_metrics import t_critical_95

IDENTITY_COLUMNS = ("run_seed", "model", "task", "split", "sequence_id")
SEQUENCE_METRICS = ("mse", "mae", "final_mse", "final_mae")


def _mean(values):
    return float(np.mean(values)) if len(values) else float("nan")


def sequence_metrics(pred, target, mask):
    """Per-sequence metric table from [N, L] arrays; unavailable quantities are NaN, never zero."""
    pred, target = np.asarray(pred, dtype=np.float64), np.asarray(target, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    rows = []
    for p, y, m in zip(pred, target, mask):
        err = p - y
        rows.append({
            "n_steps": int(m.sum()),
            "mse": _mean(err[m] ** 2),
            "mae": _mean(np.abs(err[m])),
            "final_mse": float(err[-1] ** 2) if m[-1] else float("nan"),
            "final_mae": float(abs(err[-1])) if m[-1] else float("nan"),
        })
    return pd.DataFrame(rows)


def add_identity(table, run_seed, model, task, split, sequence_ids):
    table = table.copy()
    for name, value in reversed((("run_seed", run_seed), ("model", model), ("task", task),
                                 ("split", split), ("sequence_id", np.asarray(sequence_ids)))):
        table.insert(0, name, value)
    return table


def aggregate_per_seed(seq_table, target, mask, baseline_value):
    """One row: sequence-averaged metrics plus nmse / r2 / skill against the two references.

    `target`, `mask` are the split's [N, L] arrays; `baseline_value` is the training-target mean.
    """
    target, mask = np.asarray(target, dtype=np.float64), np.asarray(mask, dtype=bool)
    ys = target[mask]
    row = {c: seq_table[c].iloc[0] for c in ("run_seed", "model", "task", "split")}
    row["n_sequences"] = len(seq_table)
    row["n_steps"] = int(seq_table["n_steps"].sum())
    for c in SEQUENCE_METRICS:
        row[c] = float(seq_table[c].mean())
    per_seq_baseline = [_mean((y[m] - baseline_value) ** 2) for y, m in zip(target, mask)]
    row["target_variance"] = float(ys.var())
    row["baseline_mse"] = float(np.mean(per_seq_baseline))
    row["nmse"] = row["mse"] / row["target_variance"] if row["target_variance"] > 0 else float("nan")
    row["r2"] = 1.0 - row["nmse"]
    row["skill"] = 1.0 - row["mse"] / row["baseline_mse"] if row["baseline_mse"] > 0 else float("nan")
    return pd.DataFrame([row])


def timestep_curve(pred, target, mask):
    """Mean squared error per timestep over supervised rows: [L] array (NaN where nothing is supervised)."""
    se = (np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64)) ** 2
    mask = np.asarray(mask, dtype=bool)
    counts = mask.sum(axis=0)
    sums = np.where(mask, se, 0.0).sum(axis=0)
    return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)


# ---------------------------------------------------------------------------------------------
# Comparisons over seeds
# ---------------------------------------------------------------------------------------------

def paired_difference(per_seed, metric, task, model_a, model_b, split="test"):
    """Per-seed paired (a - b) differences and a Student-t 95% interval over seeds."""
    sel = per_seed[(per_seed["task"] == task) & (per_seed["split"] == split)]
    a = sel[sel["model"] == model_a].set_index("run_seed")[metric]
    b = sel[sel["model"] == model_b].set_index("run_seed")[metric]
    seeds = sorted(set(a.index) & set(b.index))
    diffs = np.array([a[s] - b[s] for s in seeds], dtype=float)
    n = len(diffs)
    mean = float(diffs.mean()) if n else float("nan")
    sd = float(diffs.std(ddof=1)) if n > 1 else float("nan")
    half = t_critical_95(n - 1) * sd / math.sqrt(n) if n > 1 else float("nan")
    return {
        "task": task, "split": split, "metric": metric, "model_a": model_a, "model_b": model_b,
        "n_paired_seeds": n, "mean_difference": mean, "sd_difference": sd,
        "ci95_low": mean - half, "ci95_high": mean + half, "seeds_a_lower": int((diffs < 0).sum()),
    }


def model_summary(per_seed, metrics, split="test"):
    """Mean and sample standard deviation across seeds for each task/model/metric."""
    sel = per_seed[per_seed["split"] == split]
    rows = []
    for (task, model), group in sel.groupby(["task", "model"]):
        row = {"task": task, "model": model, "n_seeds": int(group["run_seed"].nunique())}
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            values = values[~np.isnan(values)]
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else float("nan")
            row[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)
