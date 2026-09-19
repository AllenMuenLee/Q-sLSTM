# Metrics for the nearest-neighbor memory-revision experiment.
#
# Every metric uses only rows where `metric_mask` is true (second candidate onward): the reference
# token and the first candidate never contribute. Metrics needing pred_(t-1) additionally require
# metric_mask[t-1], so t = 2 (whose predecessor is the first candidate) is skipped for them.
# Sequences are summarized first; per-seed tables then average sequences (never timesteps).

from __future__ import annotations

import math

import numpy as np
import pandas as pd

LAGS = (-2, -1, 0, 1, 2, 3, 4, 6, 8, 12, 16)
CASE_ORDER = ("iid", "late", "early", "near_best")
COUNT_COLUMNS = ("n_sequences", "n_metric_steps", "n_events", "n_nonevents", "n_drift_steps", "n_revision_steps")
IDENTITY_COLUMNS = ("run_seed", "model", "split", "case_type", "sequence_id")


def lag_name(lag):
    return f"m{-lag}" if lag < 0 else str(lag)


def _mean(values):
    return float(np.mean(values)) if len(values) else float("nan")


def _at_lag(values, valid, steps, lag):
    """Mean of `values` at step + lag over the given steps whose shifted step is a metric step."""
    shifted = steps + lag
    shifted = shifted[(shifted >= 0) & (shifted < len(values))]
    shifted = shifted[valid[shifted]]
    return _mean(values[shifted])


def sequence_metrics(pred, target, event_mask, metric_mask, alpha=None, lags=LAGS):
    """Per-sequence metric table. Inputs are [N, L] arrays; returns one row per sequence.

    Unavailable quantities (no events, no eligible lag, ...) are NaN, never zero.
    """
    pred, target = np.asarray(pred, dtype=np.float64), np.asarray(target, dtype=np.float64)
    event_mask, metric_mask = np.asarray(event_mask, dtype=bool), np.asarray(metric_mask, dtype=bool)
    if alpha is not None:
        alpha = np.asarray(alpha, dtype=np.float64)
    n_seq, length = pred.shape
    rows = []
    for i in range(n_seq):
        p, y, e, m = pred[i], target[i], event_mask[i], metric_mask[i]
        err = p - y
        ae, se = np.abs(err), err**2
        events, nonevents = m & e, m & ~e

        # steps whose predecessor is also a metric step (t >= 3 given the first-candidate exclusion)
        prev_ok = np.zeros(length, dtype=bool)
        prev_ok[1:] = m[1:] & m[:-1]
        drift = np.zeros(length)
        drift[1:] = np.abs(p[1:] - p[:-1])
        gain = np.zeros(length)
        gain[1:] = np.abs(p[:-1] - y[1:]) - ae[1:]

        best_event = int(np.flatnonzero(e)[-1]) if e.any() else -1  # global-best event = last record
        retention = prev_ok & ~e & (np.arange(length) > best_event)

        row = {
            "n_metric_steps": int(m.sum()),
            "n_events": int(events.sum()),
            "n_nonevents": int(nonevents.sum()),
            "n_drift_steps": int((prev_ok & ~e).sum()),
            "n_revision_steps": int((prev_ok & e).sum()),
            "mse": _mean(se[m]),
            "mae": _mean(ae[m]),
            "final_mse": float(se[-1]) if m[-1] else float("nan"),
            "final_mae": float(ae[-1]) if m[-1] else float("nan"),
            "event_mse": _mean(se[events]),
            "event_mae": _mean(ae[events]),
            "nonevent_mse": _mean(se[nonevents]),
            "nonevent_mae": _mean(ae[nonevents]),
            "drift_nonevent": _mean(drift[prev_ok & ~e]),
            "drift_post_best": _mean(drift[retention]),
            "revision_gain": _mean(gain[prev_ok & e]),
        }
        event_steps = np.flatnonzero(events)
        best_steps = np.array([best_event]) if best_event >= 0 and m[best_event] else np.array([], dtype=int)
        for lag in lags:
            name = lag_name(lag)
            row[f"event_abs_err_lag_{name}"] = _at_lag(ae, m, event_steps, lag)
            row[f"best_abs_err_lag_{name}"] = _at_lag(ae, m, best_steps, lag)
        if alpha is not None:
            a = alpha[i]
            row["alpha_event"] = _mean(a[events])
            row["alpha_nonevent"] = _mean(a[nonevents])
            row["alpha_contrast"] = (
                row["alpha_event"] - row["alpha_nonevent"]
                if events.any() and nonevents.any() else float("nan")
            )
            for lag in lags:
                name = lag_name(lag)
                row[f"event_alpha_lag_{name}"] = _at_lag(a, m, event_steps, lag)
                row[f"best_alpha_lag_{name}"] = _at_lag(a, m, best_steps, lag)
        rows.append(row)
    return pd.DataFrame(rows)


def add_identity(table, run_seed, model, split, case_types, sequence_ids):
    table = table.copy()
    table.insert(0, "sequence_id", np.asarray(sequence_ids))
    table.insert(0, "case_type", list(case_types))
    table.insert(0, "split", split)
    table.insert(0, "model", model)
    table.insert(0, "run_seed", run_seed)
    return table


def aggregate_per_seed(seq_table, group_columns=("run_seed", "model", "split")):
    """Average sequence metrics within (seed, model, split, case_type), plus an 'all' pooling.

    Count columns are summed. For every other metric, `<metric>__n` gives the number of sequences
    that contributed (sequences where the quantity is undefined are excluded, not zero-filled).
    """
    group_columns = list(group_columns)
    seq_table = seq_table.copy()
    seq_table["n_sequences"] = 1
    metric_columns = [c for c in seq_table.columns if c not in IDENTITY_COLUMNS and c not in COUNT_COLUMNS]
    stacked = pd.concat([seq_table, seq_table.assign(case_type="all")], ignore_index=True)

    out = []
    for keys, group in stacked.groupby(group_columns + ["case_type"], sort=True):
        row = dict(zip(group_columns + ["case_type"], keys))
        for c in COUNT_COLUMNS:
            row[c] = int(group[c].sum())
        for c in metric_columns:
            values = group[c].to_numpy(dtype=float)
            valid = ~np.isnan(values)
            row[c] = float(values[valid].mean()) if valid.any() else float("nan")
            row[f"{c}__n"] = int(valid.sum())
        out.append(row)
    table = pd.DataFrame(out)
    order = {c: i for i, c in enumerate(("all",) + CASE_ORDER)}
    table["_order"] = table["case_type"].map(order)
    table = table.sort_values(group_columns + ["_order"]).drop(columns="_order")
    return table.reset_index(drop=True)


# ---------------------------------------------------------------------------------------------
# Paired comparison over seeds
# ---------------------------------------------------------------------------------------------

_T975 = {  # two-sided 95% critical values of Student's t
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
    10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110,
    18: 2.101, 19: 2.093, 20: 2.086, 25: 2.060, 30: 2.042,
}


def t_critical_95(dof):
    if dof < 1:
        return float("nan")
    if dof >= 30:
        return _T975[30] if dof < 120 else 1.98
    return _T975[max(k for k in _T975 if k <= dof)]


def paired_difference(per_seed, metric="mse", case_type="all", split="test",
                      model_a="qslstm", model_b="qlstm"):
    """Per-seed paired (a - b) differences and their summary.

    Returns (table, summary). The 95% interval is a Student-t interval over seed-level paired
    differences (n = number of paired seeds); timesteps and sequences are never treated as samples.
    """
    sel = per_seed[(per_seed["case_type"] == case_type) & (per_seed["split"] == split)]
    a = sel[sel["model"] == model_a].set_index("run_seed")[metric]
    b = sel[sel["model"] == model_b].set_index("run_seed")[metric]
    seeds = sorted(set(a.index) & set(b.index))
    table = pd.DataFrame({"run_seed": seeds, model_a: [a[s] for s in seeds], model_b: [b[s] for s in seeds]})
    table["difference"] = table[model_a] - table[model_b]
    n = len(table)
    diffs = table["difference"].to_numpy(dtype=float)
    mean = float(diffs.mean()) if n else float("nan")
    sd = float(diffs.std(ddof=1)) if n > 1 else float("nan")
    half = t_critical_95(n - 1) * sd / math.sqrt(n) if n > 1 else float("nan")
    summary = {
        "metric": metric, "case_type": case_type, "split": split,
        "model_a": model_a, "model_b": model_b, "n_paired_seeds": n,
        "mean_difference": mean, "sd_difference": sd,
        "ci95_low": mean - half, "ci95_high": mean + half,
        "ci_method": "Student-t interval on seed-level paired differences (a - b)",
        "seeds_a_lower": int((diffs < 0).sum()),
    }
    return table, summary


def model_summary(per_seed, metrics, split="test", case_types=("all",) + CASE_ORDER):
    """Mean and sample standard deviation across seeds for each model/case/metric."""
    sel = per_seed[(per_seed["split"] == split) & per_seed["case_type"].isin(case_types)]
    rows = []
    for (model, case), group in sel.groupby(["model", "case_type"]):
        row = {"model": model, "case_type": case, "n_seeds": int(group["run_seed"].nunique())}
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            values = values[~np.isnan(values)]
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else float("nan")
            row[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# Timestep curves from the tidy prediction table
# ---------------------------------------------------------------------------------------------

def timestep_curves(pred_frame):
    """Per (run_seed, model, case_type, timestep) mean squared error and non-event retention drift.

    Only metric steps are used; drift additionally needs the previous step to be a metric step.
    """
    df = pred_frame.sort_values(["run_seed", "model", "sequence_id", "timestep"]).copy()
    keys = ["run_seed", "model", "sequence_id"]
    df["prev_pred"] = df.groupby(keys)["prediction"].shift(1)
    df["prev_metric"] = df.groupby(keys)["is_metric_step"].shift(1).fillna(False).astype(bool)
    df["drift"] = (df["prediction"] - df["prev_pred"]).abs()
    df.loc[~(df["is_metric_step"] & df["prev_metric"]) | df["is_event"], "drift"] = np.nan
    metric_rows = df[df["is_metric_step"]]
    return (metric_rows.groupby(["run_seed", "model", "case_type", "timestep"])
            .agg(mse=("squared_error", "mean"), drift=("drift", "mean"), n=("squared_error", "size"))
            .reset_index())
