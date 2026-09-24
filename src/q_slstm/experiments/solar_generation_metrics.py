# Metrics for the solar-generation forecasting experiment (physical units: MWh, MWh^2).
#
# There is one prediction per target hour, so every metric is an hourly-weighted mean over target rows
# within one seed; seeds are the replication unit for paired comparisons. Missing baselines stay NaN
# and never count as zero; baselines and models are compared on the same eligible target subset.

from __future__ import annotations

import math

import numpy as np
import pandas as pd

FORECASTERS = ("model", "persistence", "daily_persistence")
FORECAST_COLUMNS = {"model": "prediction_mwh", "persistence": "persistence_mwh",
                    "daily_persistence": "daily_persistence_mwh"}
METRIC_NAMES = ("mse", "mae", "rmse", "r2")

_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
         10: 2.228, 15: 2.131, 20: 2.086, 30: 2.042}


def t_critical_95(dof):
    if dof < 1:
        return math.nan
    return _T975[max(k for k in _T975 if k <= dof)] if dof < 120 else 1.98


def point_metrics(target, prediction):
    """n, MSE (MWh^2), MAE, RMSE (MWh), R^2 (None when the target has zero variance or n == 0)."""
    y, p = np.asarray(target, dtype=float), np.asarray(prediction, dtype=float)
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch {y.shape} vs {p.shape}")
    n = len(y)
    if n == 0:
        return {"n": 0, "mse": math.nan, "mae": math.nan, "rmse": math.nan, "r2": None}
    err = p - y
    mse = float(np.mean(err**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {"n": n, "mse": mse, "mae": float(np.mean(np.abs(err))), "rmse": math.sqrt(mse),
            "r2": None if ss_tot == 0 else 1.0 - float(np.sum(err**2)) / ss_tot}


def _group_rows(frame, forecasters, subset, identity):
    rows = []
    groupings = [("overall", None), ("month", "target_month"), ("hour_est", "target_hour_est"),
                 ("daylight_proxy", "is_daylight_proxy")]
    if "is_large_ramp" in frame.columns:
        groupings.append(("large_ramp", "is_large_ramp"))
    for group_type, column in groupings:
        if column is None:
            groups = [("all", frame)]
        else:
            key = frame[column].astype(object).where(frame[column].notna(), "missing").astype(str)
            groups = list(frame.groupby(key, sort=True))
        for value, g in groups:
            for f in forecasters:
                m = point_metrics(g["target_mwh"], g[FORECAST_COLUMNS[f]])
                row = {**identity, "subset": subset, "group_type": group_type, "group": value, "forecaster": f, **m}
                if f == "model":
                    neg = int((g["prediction_mwh"] < 0).sum())
                    row.update(n_negative_predictions=neg, negative_prediction_fraction=neg / len(g) if len(g) else math.nan)
                rows.append(row)
    return rows


def seed_metrics(pred):
    """Long per-seed metric table from one run's predictions_test.csv.

    subset `all_eligible`: every eligible test target (model and hourly persistence, which is always
    available because the origin row is valid). subset `baseline_common`: targets where daily
    persistence is also available, for a like-for-like comparison of all three forecasters.
    Daylight groups include `missing` when target-hour GHI is unavailable.
    """
    identity = {k: pred[k].iloc[0] for k in ("run_seed", "model", "dataset_id")}
    rows = _group_rows(pred, ("model", "persistence"), "all_eligible", identity)
    common = pred[pred["daily_persistence_mwh"].notna()]
    rows += _group_rows(common, FORECASTERS, "baseline_common", identity)
    return pd.DataFrame(rows)


def daily_metrics(pred):
    """Per EST target date metrics (for temporal inspection, not the primary score)."""
    rows = []
    for date, g in pred.groupby("target_date_est", sort=True):
        row = {"run_seed": g["run_seed"].iloc[0], "model": g["model"].iloc[0], "target_date_est": date}
        for f in ("model", "persistence"):
            m = point_metrics(g["target_mwh"], g[FORECAST_COLUMNS[f]])
            row.update({f"{f}_{k}": m[k] for k in ("n", "mse", "mae")})
        rows.append(row)
    return pd.DataFrame(rows)


def alpha_curves(alpha_steps, daylight, identity):
    """Mean write proportion by input step (step 0 excluded), for all targets and daylight groups.

    `alpha_steps`: [n_windows, L] hidden-mean alpha; `daylight`: nullable boolean per window.
    """
    alpha_steps = np.asarray(alpha_steps, dtype=float)
    day = pd.array(daylight, dtype="boolean")
    groups = {"all": np.ones(len(alpha_steps), dtype=bool),
              "daylight": np.asarray(day.fillna(False), dtype=bool),
              "night": np.asarray((~day).fillna(False), dtype=bool)}
    rows = []
    for name, mask in groups.items():
        for step in range(1, alpha_steps.shape[1]):
            vals = alpha_steps[mask, step]
            rows.append({**identity, "target_group": name, "input_step": step,
                         "steps_before_origin": alpha_steps.shape[1] - 1 - step,
                         "alpha_mean": float(vals.mean()) if len(vals) else math.nan, "n_windows": int(len(vals))})
    return pd.DataFrame(rows)


def paired_difference(per_seed, metric="mse", subset="all_eligible", group_type="overall", group="all",
                      forecaster="model", model_a="qslstm", model_b="qlstm"):
    """Seed-level (a - b) differences and a Student-t 95% interval over paired seeds."""
    sel = per_seed[(per_seed["subset"] == subset) & (per_seed["group_type"] == group_type)
                   & (per_seed["group"].astype(str) == str(group)) & (per_seed["forecaster"] == forecaster)]
    a = sel[sel["model"] == model_a].set_index("run_seed")[metric]
    b = sel[sel["model"] == model_b].set_index("run_seed")[metric]
    seeds = sorted(set(a.index) & set(b.index))
    table = pd.DataFrame({"run_seed": seeds, model_a: [float(a[s]) for s in seeds],
                          model_b: [float(b[s]) for s in seeds]})
    table["difference"] = table[model_a] - table[model_b]
    n = len(table)
    diffs = table["difference"].to_numpy(dtype=float)
    mean = float(diffs.mean()) if n else math.nan
    sd = float(diffs.std(ddof=1)) if n > 1 else math.nan
    half = t_critical_95(n - 1) * sd / math.sqrt(n) if n > 1 else math.nan
    summary = {"metric": metric, "subset": subset, "group_type": group_type, "group": group,
               "model_a": model_a, "model_b": model_b, "n_paired_seeds": n,
               "mean_difference": mean, "sd_difference": sd, "ci95_low": mean - half, "ci95_high": mean + half,
               "ci_method": "Student-t interval over seed-level paired differences (training randomness on a "
                            "fixed period; not uncertainty across years)",
               "seeds_a_lower": int((diffs < 0).sum())}
    return table, summary
