# Hourly table, forecast windows, chronological splits, scaler, and the frozen processed dataset.
#
# Row i of the hourly table is the IESO interval (timestamp_utc - 1h, timestamp_utc]; the complete
# expected grid is built before any filtering, so gaps never compress time. A window with origin t
# uses rows t-L+1..t as inputs ([L, 13]) and predicts solar_generation at row t+1. Windows are
# assigned to splits by target row, and a window is eligible only when every row it touches (inputs
# and target) is valid. Only the 13 FEATURE_COLUMNS ever reach the model.

from __future__ import annotations

import datetime as dt
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from q_slstm.data_sources import ieso, open_meteo
from q_slstm.data_sources.cache import SnapshotStore, atomic_write_json, canonical_json, sha256_bytes, sha256_file

SCHEMA_VERSION = "solar_generation_dataset/1"
WEATHER_COLUMNS = open_meteo.WEATHER_VARIABLES
PHYSICAL_COLUMNS = ("solar_generation",) + WEATHER_COLUMNS
CYCLIC_COLUMNS = ("hour_sin", "hour_cos", "day_sin", "day_cos")
FEATURE_COLUMNS = PHYSICAL_COLUMNS + CYCLIC_COLUMNS
RAW_INPUT_SIZE = len(FEATURE_COLUMNS)  # 13
TARGET_COLUMN = "solar_generation"
FEATURE_UNITS = {
    "solar_generation": "MWh per hourly interval", **{k: v for k, v in open_meteo.EXPECTED_UNITS.items() if k != "time"},
    "precipitation": "mm per hourly interval",
    **{c: "dimensionless [-1, 1]" for c in CYCLIC_COLUMNS},
}
SPLIT_NAMES = ("train", "val", "test")
DEFAULT_MIN_COVERAGE = 0.95
DAYLIGHT_GHI_THRESHOLD = 20.0  # W/m^2, aggregated target-hour GHI; grouping only, never an input
RAMP_PERCENTILE = 90.0
TRACE_BLOCK_HOURS = 7 * 24
SEQUENCE_LENGTH = {"paper": 32, "pilot": 8}
TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

QUALITY_POLICY = {
    "imputation": "none",
    "generation_missing": "row invalid (absent SOLAR record, EnergyValue, or Output is missing, never zero)",
    "generation_quality_flag": "row invalid when IESO OutputQuality != 0 (telemetry data points unavailable)",
    "generation_negative": "row invalid (quarantined, not clipped)",
    "weather_missing": "row invalid when any configured location lacks a value (weights never renormalized)",
    "weather_ranges": "radiation, wind speed, precipitation >= 0; cloud cover and humidity in [0, 100]; "
                      "temperature finite; violations quarantined, not clipped",
    "window_rule": "a window is eligible only if all L input rows and the target row are valid",
    "nighttime_zeros": "valid observations",
}


class DatasetQualityError(RuntimeError):
    """Coverage or window requirements not met; the quality report explains why."""


def fmt_ts(series):
    return pd.DatetimeIndex(series).strftime(TS_FORMAT)


# ---------------------------------------------------------------------------------------------
# Grid and calendar
# ---------------------------------------------------------------------------------------------

def expected_grid(start_date, end_date):
    """Every (source_date, hour ending) of the inclusive EST date range with its UTC interval."""
    days = pd.date_range(start_date, end_date, freq="D")
    source_date = np.repeat(days.strftime("%Y-%m-%d"), 24)
    source_hour = np.tile(np.arange(1, 25), len(days))
    first_end = ieso.interval_end_utc(pd.Timestamp(start_date).date(), 1)
    ends = pd.date_range(first_end, periods=len(source_hour), freq="h")
    return pd.DataFrame({"timestamp_utc": ends, "interval_start_utc": ends - pd.Timedelta(hours=1),
                         "source_date": source_date, "source_hour": source_hour})


def calendar_features(interval_start_utc):
    """Cyclic hour-of-day and day-of-year from the interval START in fixed EST (hour labels 0..23).

    day_* uses (d - 1) / D_year with D_year = 366 in leap years and 365 otherwise.
    """
    local = pd.DatetimeIndex(interval_start_utc).tz_convert(ieso.EST)
    hour = local.hour.to_numpy(dtype=float)
    doy = local.dayofyear.to_numpy(dtype=float)
    days_in_year = np.where(local.is_leap_year, 366.0, 365.0)
    return pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * hour / 24), "hour_cos": np.cos(2 * np.pi * hour / 24),
        "day_sin": np.sin(2 * np.pi * (doy - 1) / days_in_year),
        "day_cos": np.cos(2 * np.pi * (doy - 1) / days_in_year),
    })


def split_boundaries(n):
    """Row indices (train_end, val_end) = (floor(0.7 n), floor(0.8 n)) in exact integer arithmetic."""
    return (n * 7) // 10, (n * 8) // 10


def row_split(n):
    train_end, val_end = split_boundaries(n)
    labels = np.empty(n, dtype=object)
    labels[:train_end], labels[train_end:val_end], labels[val_end:] = "train", "val", "test"
    return labels


# ---------------------------------------------------------------------------------------------
# Hourly table
# ---------------------------------------------------------------------------------------------

def build_hourly_table(grid, generation, weather, weather_reasons):
    """Join generation and aggregated weather onto the complete grid and mark row validity.

    `generation`: parsed IESO frame (source_date, source_hour, solar_generation, output_quality,
    status, ...). `weather`: aggregate frame aligned row-for-row with `grid`.
    """
    if generation.duplicated(["source_date", "source_hour"]).any():
        raise ValueError("duplicate generation records reached build_hourly_table")
    gen = generation.rename(columns={"timestamp_utc": "gen_timestamp_utc"}).drop(
        columns=["interval_start_utc", "source_file"], errors="ignore")
    table = grid.merge(gen, on=["source_date", "source_hour"], how="left", validate="one_to_one")
    if "gen_timestamp_utc" in table:
        present = table["gen_timestamp_utc"].notna()
        if not (table.loc[present, "gen_timestamp_utc"] == table.loc[present, "timestamp_utc"]).all():
            raise ValueError("generation interval ends disagree with the expected grid")
        table = table.drop(columns="gen_timestamp_utc")
    table["status"] = table["status"].fillna("missing")
    table["fuel_record_present"] = table["fuel_record_present"].astype("boolean").fillna(False)
    if not (weather["timestamp_utc"].to_numpy() == table["timestamp_utc"].to_numpy()).all():
        raise ValueError("weather frame is not aligned with the grid")
    for name in WEATHER_COLUMNS:
        table[name] = weather[name].to_numpy()
    cal = calendar_features(table["interval_start_utc"])
    for name in CYCLIC_COLUMNS:
        table[name] = cal[name].to_numpy()

    reasons = []
    for i, row in enumerate(table.itertuples(index=False)):
        r = []
        if row.status == "missing":
            r.append("generation_missing" if row.fuel_record_present else "generation_record_absent")
        elif row.status == "flagged":
            r.append(f"generation_quality_flag_{row.output_quality}")
        if row.status != "missing" and not (row.solar_generation >= 0):
            r.append("solar_generation_negative" if np.isfinite(row.solar_generation) else "solar_generation_non_finite")
        r += weather_reasons[i]
        reasons.append(r)
    table["invalid_reasons"] = [";".join(r) for r in reasons]
    table["row_valid"] = [not r for r in reasons]
    table["row_split"] = row_split(len(table))
    table = table.rename(columns={"status": "generation_status"})
    return table


# ---------------------------------------------------------------------------------------------
# Windows, scaler, thresholds
# ---------------------------------------------------------------------------------------------

def build_windows(table, sequence_length, ramp_threshold=None):
    """Every candidate window (target index L..N-1) with eligibility and analysis metadata."""
    n, L = len(table), sequence_length
    if L < 1:
        raise ValueError("sequence_length must be positive")
    valid = table["row_valid"].to_numpy(dtype=bool)
    # invalid_count[k] = invalid rows among 0..k-1, so rows a..b inclusive contain c[b+1]-c[a]
    invalid_count = np.concatenate([[0], np.cumsum(~valid)])
    targets = np.arange(L, n)
    touched_invalid = invalid_count[targets + 1] - invalid_count[targets - L]
    y = table[TARGET_COLUMN].to_numpy(dtype=float)
    ghi = table["shortwave_radiation"].to_numpy(dtype=float)
    local = pd.DatetimeIndex(table["interval_start_utc"]).tz_convert(ieso.EST)

    daily = np.full(len(targets), np.nan)
    has_day = targets >= 24
    lag_idx = targets[has_day] - 24
    daily[has_day] = np.where(valid[lag_idx], y[lag_idx], np.nan)  # missing history stays missing
    daylight = np.where(np.isfinite(ghi[targets]), (ghi[targets] > DAYLIGHT_GHI_THRESHOLD).astype(float), np.nan)

    windows = pd.DataFrame({
        "window_id": targets,
        "origin_index": targets - 1,
        "target_index": targets,
        "origin_timestamp_utc": fmt_ts(table["timestamp_utc"].iloc[targets - 1]),
        "target_timestamp_utc": fmt_ts(table["timestamp_utc"].iloc[targets]),
        "split": table["row_split"].to_numpy()[targets],
        "eligible": touched_invalid == 0,
        "n_invalid_rows_touched": touched_invalid,
        "target_mwh": y[targets],
        "persistence_mwh": y[targets - 1],
        "daily_persistence_mwh": daily,
        "target_month": local.month.to_numpy()[targets],
        "target_hour_est": local.hour.to_numpy()[targets],
        "target_date_est": local.strftime("%Y-%m-%d").to_numpy()[targets],
        "is_daylight_proxy": pd.array(daylight, dtype="Float64").astype("boolean"),
    })
    if ramp_threshold is not None:
        ramp = np.abs(y[targets] - y[targets - 1])
        windows["abs_ramp_mwh"] = ramp
        windows["is_large_ramp"] = ramp >= ramp_threshold
    return windows


def fit_scaler(table):
    """Per-column mean / std (ddof=0) of the physical columns over unique valid training rows."""
    fit = table[(table["row_split"] == "train") & table["row_valid"]]
    if fit.empty:
        raise DatasetQualityError("no valid training rows to fit the scaler")
    columns = {}
    for name in PHYSICAL_COLUMNS:
        values = fit[name].to_numpy(dtype=float)
        std = float(values.std(ddof=0))
        columns[name] = {"mean": float(values.mean()), "scale": std if std > 0 else 1.0,
                         "std": std, "zero_variance": bool(std == 0)}
    for name in CYCLIC_COLUMNS:
        columns[name] = {"mean": 0.0, "scale": 1.0, "std": None, "zero_variance": False, "unscaled": True}
    fit_idx = fit.index.to_numpy(dtype="<i8")
    return {
        "method": "standardize physical columns with training mean and population std (ddof=0); "
                  "cyclic columns unchanged; zero variance uses scale 1",
        "feature_order": list(FEATURE_COLUMNS),
        "target_column": TARGET_COLUMN,
        "target": {"mean": columns[TARGET_COLUMN]["mean"], "scale": columns[TARGET_COLUMN]["scale"]},
        "columns": columns,
        "n_fit_rows": int(len(fit)),
        "fit_row_index_sha256": sha256_bytes(fit_idx.tobytes()),
    }


def apply_scaler(table, scaler):
    """[N, 13] float32 scaled features in FEATURE_COLUMNS order (invalid rows keep NaN)."""
    out = np.empty((len(table), len(FEATURE_COLUMNS)), dtype=np.float64)
    for j, name in enumerate(FEATURE_COLUMNS):
        c = scaler["columns"][name]
        out[:, j] = (table[name].to_numpy(dtype=float) - c["mean"]) / c["scale"]
    return out.astype(np.float32)


def scale_target(values, scaler):
    return (np.asarray(values, dtype=float) - scaler["target"]["mean"]) / scaler["target"]["scale"]


def unscale_target(values, scaler):
    return np.asarray(values, dtype=float) * scaler["target"]["scale"] + scaler["target"]["mean"]


def ramp_threshold(table, percentile=RAMP_PERCENTILE):
    """Percentile of |y_t - y_(t-1)| over consecutive valid training rows (training data only)."""
    y = table[TARGET_COLUMN].to_numpy(dtype=float)
    ok = (table["row_valid"] & (table["row_split"] == "train")).to_numpy(dtype=bool)
    pair = ok[1:] & ok[:-1]
    diffs = np.abs(np.diff(y))[pair]
    if len(diffs) == 0:
        return None, 0
    return float(np.percentile(diffs, percentile)), int(len(diffs))


def trace_block(windows, table):
    """First complete seven-day block of eligible test windows starting at an EST midnight.

    Chosen from eligibility alone (before any prediction exists). Falls back to the first contiguous
    run of eligible test windows when no complete block exists (e.g. the short pilot range).
    """
    test = windows[(windows["split"] == "test")].set_index("target_index")
    eligible = test["eligible"]
    idx = test.index.to_numpy()
    starts = test.index[test["target_hour_est"] == 0]
    for s in starts:
        block = np.arange(s, s + TRACE_BLOCK_HOURS)
        if block[-1] in eligible.index and eligible.reindex(block).fillna(False).all():
            return _block_record(test, block, complete=True)
    runs, current = [], []
    for i in idx:
        if eligible[i] and (not current or i == current[-1] + 1):
            current.append(i)
        else:
            if current:
                runs.append(current)
            current = [i] if eligible[i] else []
    if current:
        runs.append(current)
    if not runs:
        return {"complete_7_day_block": False, "n_hours": 0, "window_ids": []}
    return _block_record(test, np.array(runs[0][:TRACE_BLOCK_HOURS]), complete=False)


def _block_record(test, block, complete):
    return {
        "complete_7_day_block": complete,
        "rule": "first seven consecutive days of eligible test windows starting at 00:00 EST"
                + ("" if complete else "; none exists, so the first contiguous eligible test run is used"),
        "n_hours": int(len(block)),
        "first_target_timestamp_utc": test.loc[block[0], "target_timestamp_utc"],
        "last_target_timestamp_utc": test.loc[block[-1], "target_timestamp_utc"],
        "window_ids": [int(x) for x in test.loc[block, "window_id"]],
    }


def alignment_diagnostic(table, lags=(-2, -1, 0, 1, 2)):
    """Correlation of generation with aggregated GHI shifted by `lag` hours, by DST season.

    Audits the fixed-EST interval mapping: with the correct join the peak is at lag 0 in both
    seasons; a daylight-saving mistake would move the summer peak by one hour.
    """
    y = table[TARGET_COLUMN].to_numpy(dtype=float)
    ghi = table["shortwave_radiation"].to_numpy(dtype=float)
    valid = table["row_valid"].to_numpy(dtype=bool)
    month = pd.DatetimeIndex(table["interval_start_utc"]).tz_convert(ieso.EST).month.to_numpy()
    seasons = {"apr_oct": (month >= 4) & (month <= 10), "nov_mar": (month <= 3) | (month >= 11)}
    out = {}
    for season, mask in seasons.items():
        row = {}
        for lag in lags:
            i = np.arange(max(0, -lag), min(len(y), len(y) - lag))
            ok = valid[i] & valid[i + lag] & mask[i]
            row[str(lag)] = float(np.corrcoef(y[i][ok], ghi[i + lag][ok])[0, 1]) if ok.sum() > 2 else None
        best = max((k for k in row if row[k] is not None), key=lambda k: row[k], default=None)
        out[season] = {"correlation_by_weather_lag_hours": row, "best_lag": None if best is None else int(best),
                       "n_rows": int(mask.sum())}
    return out


# ---------------------------------------------------------------------------------------------
# Torch dataset
# ---------------------------------------------------------------------------------------------

class SolarWindowDataset(Dataset):
    """Items: inputs [L, 13] float32, target [1] float32 (scaled), plus identifying metadata.

    Only `inputs` is a model input; the other fields identify the window for analysis.
    """

    def __init__(self, features, targets_scaled, windows, sequence_length, split):
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.targets = torch.as_tensor(targets_scaled, dtype=torch.float32)
        self.windows = windows.reset_index(drop=True)
        self.sequence_length = sequence_length
        self.split = split
        self.target_index = self.windows["target_index"].to_numpy(dtype=np.int64)
        self.window_ids = self.windows["window_id"].to_numpy(dtype=np.int64)
        if len(self.windows) and not self.windows["eligible"].all():
            raise ValueError("SolarWindowDataset received ineligible windows")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, k):
        j = int(self.target_index[k])
        return {
            "inputs": self.features[j - self.sequence_length:j],
            "target": self.targets[j:j + 1],
            "window_id": int(self.window_ids[k]),
            "origin_timestamp_utc": self.windows.at[k, "origin_timestamp_utc"],
            "target_timestamp_utc": self.windows.at[k, "target_timestamp_utc"],
            "split": self.split,
        }


# ---------------------------------------------------------------------------------------------
# Preparation
# ---------------------------------------------------------------------------------------------

def _load_generation(store, collection, start, end):
    frames = []
    for record in collection["ieso"]["files"]:
        data = store.load(record)
        _, frame = ieso.parse_generation_xml(data, record["filename"])
        frame["source_file"] = record["filename"]
        frames.append(frame)
    gen = pd.concat(frames, ignore_index=True)
    gen = gen[(gen["source_date"] >= str(start)) & (gen["source_date"] <= str(end))]
    dups = gen.duplicated(["source_date", "source_hour"], keep=False)
    if dups.any():
        raise ieso.IesoSchemaError(f"generation records for the same hour appear in several files: "
                                   f"{gen[dups].head().to_dict('records')}")
    return ieso.normalize_generation(gen.reset_index(drop=True))


def _load_weather(store, collection, grid_ts):
    per_location, audit = {}, {}
    for loc, records in collection["open_meteo"].items():
        frames = []
        for record in records:
            payload = json.loads(store.load(record))
            _, frame = open_meteo.parse_archive_json(payload, record["params"], record["key"])
            frames.append(frame)
        combined, collapsed = open_meteo.combine_chunks(frames)
        trimmed = combined[combined["timestamp_utc"].isin(grid_ts)].reset_index(drop=True)
        missing_ts = int(len(grid_ts) - len(trimmed))
        per_location[loc] = trimmed
        audit[loc] = {"n_requests": len(records), "boundary_duplicates_collapsed": collapsed,
                      "rows_after_trim": int(len(trimmed)), "grid_timestamps_absent": missing_ts,
                      "non_finite_values": int((~np.isfinite(trimmed[list(WEATHER_COLUMNS)].to_numpy())).sum())}
    return per_location, audit


def _csv_bytes(frame):
    return frame.to_csv(index=False, float_format="%.10g", lineterminator="\n").encode("utf-8")


def _coverage(table, windows):
    report = {}
    for split in SPLIT_NAMES:
        rows = table[table["row_split"] == split]
        win = windows[windows["split"] == split]
        report[split] = {
            "expected_rows": int(len(rows)),
            "valid_rows": int(rows["row_valid"].sum()),
            "valid_row_fraction": float(rows["row_valid"].mean()) if len(rows) else 0.0,
            "candidate_windows": int(len(win)),
            "eligible_windows": int(win["eligible"].sum()),
            "valid_window_fraction": float(win["eligible"].mean()) if len(win) else 0.0,
            "first_timestamp_utc": fmt_ts(rows["timestamp_utc"].iloc[[0]])[0] if len(rows) else None,
            "last_timestamp_utc": fmt_ts(rows["timestamp_utc"].iloc[[-1]])[0] if len(rows) else None,
        }
    return report


def _reason_counts(table):
    counts = {}
    for text in table["invalid_reasons"]:
        for r in filter(None, text.split(";")):
            key = r.split(":", 1)[-1] if ":" in r else r
            counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def prepare_dataset(data_dir, scale, collection_file=None, sequence_length=None, min_coverage=DEFAULT_MIN_COVERAGE,
                    log=print):
    """Build (or reuse) the frozen processed dataset for a collection; returns (manifest, pointer path).

    Reads only pinned raw snapshots (no network). Fails with DatasetQualityError, after writing the
    quality report, when a split is under `min_coverage` valid rows or has no eligible window.
    """
    data_dir = Path(data_dir)
    collection_file = Path(collection_file) if collection_file else data_dir / f"collection_{scale}.json"
    if not collection_file.exists():
        raise FileNotFoundError(f"{collection_file} not found; run collect_data.py --scale {scale} first "
                                "(or pass --collection for an overridden collection)")
    collection = json.loads(collection_file.read_text(encoding="utf-8"))
    if collection["scale"] != scale:
        raise ValueError(f"collection {collection_file} is for scale {collection['scale']!r}, not {scale!r}")
    overrides = dict(collection.get("date_overrides", {}))
    preset_length = SEQUENCE_LENGTH[scale]
    L = preset_length if sequence_length is None else int(sequence_length)
    if L != preset_length:
        overrides["sequence_length"] = {"preset": preset_length, "used": L}
    if not 0 < min_coverage <= 1:
        raise ValueError("min_coverage must be in (0, 1]")
    if min_coverage != DEFAULT_MIN_COVERAGE:
        overrides["min_coverage"] = {"preset": DEFAULT_MIN_COVERAGE, "used": min_coverage}
    scale_label = scale if not overrides else f"{scale}-overridden"

    start, end = dt.date.fromisoformat(collection["start_date"]), dt.date.fromisoformat(collection["end_date"])
    store = SnapshotStore(data_dir / "raw")
    locations = pd.DataFrame(collection["locations"])
    weights = dict(zip(locations["location_id"], locations["weight"]))
    log(f"preparing {scale_label}: {start}..{end}, L={L}")

    grid = expected_grid(start, end)
    generation = _load_generation(store, collection, start, end)
    per_location, weather_audit = _load_weather(store, collection, grid["timestamp_utc"])
    weather, weather_reasons = open_meteo.aggregate_locations(per_location, weights, grid["timestamp_utc"])
    table = build_hourly_table(grid, generation, weather, weather_reasons)

    threshold, n_ramp_pairs = ramp_threshold(table)
    windows = build_windows(table, L, threshold)
    scaler = fit_scaler(table)
    coverage = _coverage(table, windows)
    failures = [f"{s}: valid rows {c['valid_row_fraction']:.4f} < {min_coverage}" for s, c in coverage.items()
                if c["valid_row_fraction"] < min_coverage]
    failures += [f"{s}: no eligible window" for s, c in coverage.items() if c["eligible_windows"] == 0]

    quality = {
        "scale_label": scale_label,
        "expected_rows": int(len(table)),
        "valid_rows": int(table["row_valid"].sum()),
        "generation_status_counts": table["generation_status"].value_counts().to_dict(),
        "generation_quality_flag_counts": {str(k): int(v) for k, v in
                                           table["output_quality"].value_counts(dropna=False).items()},
        "invalid_reason_counts": _reason_counts(table),
        "duplicate_generation_records_collapsed": int(sum(r["doc_header"].get("duplicates_collapsed", 0)
                                                          for r in collection["ieso"]["files"])),
        "weather_per_location": weather_audit,
        "coverage_by_split": coverage,
        "min_coverage": min_coverage,
        "insufficient_history_targets": int(L),
        "alignment_diagnostic": alignment_diagnostic(table),
        "failures": failures,
    }
    if failures:
        report_path = data_dir / f"quality_report_{scale_label}_FAILED.json"
        atomic_write_json(report_path, quality)
        raise DatasetQualityError(f"dataset preparation failed ({report_path}): {failures}")

    tmp = Path(tempfile.mkdtemp(prefix=".prepare_", dir=_ensure(data_dir / "processed")))
    try:
        files = {}

        def put(name, data):
            (tmp / name).parent.mkdir(parents=True, exist_ok=True)
            (tmp / name).write_bytes(data)
            files[name] = sha256_bytes(data)

        gen_out = generation.copy()
        gen_out["timestamp_utc"] = fmt_ts(gen_out["timestamp_utc"])
        gen_out["interval_start_utc"] = fmt_ts(gen_out["interval_start_utc"])
        put("normalized/ieso_solar_generation.csv", _csv_bytes(gen_out))
        for loc, frame in sorted(per_location.items()):
            out = frame.copy()
            out["timestamp_utc"] = fmt_ts(out["timestamp_utc"])
            put(f"normalized/open_meteo_{loc}.csv", _csv_bytes(out))
        hourly = table.copy()
        hourly["timestamp_utc"] = fmt_ts(hourly["timestamp_utc"])
        hourly["interval_start_utc"] = fmt_ts(hourly["interval_start_utc"])
        hourly = hourly[["timestamp_utc", "interval_start_utc", "source_date", "source_hour", "row_split",
                         *FEATURE_COLUMNS, "output_quality", "generation_status", "row_valid", "invalid_reasons"]]
        put("hourly_table.csv", _csv_bytes(hourly))
        put("quarantine.csv", _csv_bytes(hourly[~hourly["row_valid"]]))
        put("windows.csv", _csv_bytes(windows))
        put("scaler.json", (json.dumps(scaler, indent=2, sort_keys=True) + "\n").encode("utf-8"))

        train_end, val_end = split_boundaries(len(table))
        ts = fmt_ts(table["timestamp_utc"])
        preparation = {
            "scale": scale, "scale_label": scale_label, "overrides": overrides,
            "start_date": str(start), "end_date": str(end), "sequence_length": L, "horizon_hours": 1,
            "stride_hours": 1, "split_fractions": {"train": 0.7, "val": 0.1, "test": 0.2},
            "min_coverage": min_coverage, "daylight_ghi_threshold_w_m2": DAYLIGHT_GHI_THRESHOLD,
            "ramp_percentile": RAMP_PERCENTILE,
        }
        identity = {
            "schema_version": SCHEMA_VERSION,
            "raw_sha256": sorted([r["sha256"] for r in collection["ieso"]["files"]] +
                                 [r["sha256"] for recs in collection["open_meteo"].values() for r in recs]),
            "locations": collection["locations"],
            "feature_order": list(FEATURE_COLUMNS),
            "units": FEATURE_UNITS,
            "quality_policy": QUALITY_POLICY,
            "preparation": preparation,
            "processed_sha256": files,
        }
        dataset_id = sha256_bytes(canonical_json(identity).encode("utf-8"))[:16]
        manifest = {
            "kind": "solar_generation_dataset",
            "dataset_id": dataset_id,
            "identity_method": "sha256 of canonical JSON of the 'identity' block (no paths or times); first 16 hex",
            "identity": identity,
            "scale_label": scale_label,
            "sequence_length": L,
            "raw_input_size": RAW_INPUT_SIZE,
            "feature_columns": list(FEATURE_COLUMNS),
            "physical_columns": list(PHYSICAL_COLUMNS),
            "cyclic_columns": list(CYCLIC_COLUMNS),
            "target": {"column": TARGET_COLUMN, "unit": "MWh", "horizon_hours": 1},
            "calendar_convention": "hour/day from the interval start in fixed EST (UTC-05:00); hour labels "
                                   "0..23; day_* = 2*pi*(day_of_year - 1)/D_year with D_year 366 in leap years, "
                                   "365 otherwise",
            "timestamp_convention": "timestamp_utc is the interval END (UTC); the row covers "
                                    "(timestamp_utc - 1h, timestamp_utc]; IESO date/hour-ending map via fixed "
                                    "UTC-05:00 without daylight saving",
            "weather_join": "Open-Meteo value stamped T (UTC) joins the interval ending at T; radiation and "
                            "precipitation describe the preceding hour, other variables the endpoint",
            "n_rows": int(len(table)),
            "split_boundaries": {
                "train": {"rows": [0, train_end], "first_timestamp_utc": ts[0], "last_timestamp_utc": ts[train_end - 1]},
                "val": {"rows": [train_end, val_end], "first_timestamp_utc": ts[train_end],
                        "last_timestamp_utc": ts[val_end - 1]},
                "test": {"rows": [val_end, len(table)], "first_timestamp_utc": ts[val_end],
                         "last_timestamp_utc": ts[len(table) - 1]},
                "rule": "floor(0.7 N), floor(0.8 N) over the complete expected grid, before quality filtering; "
                        "windows are assigned by target row",
            },
            "window_counts": {s: {"candidate": c["candidate_windows"], "eligible": c["eligible_windows"]}
                              for s, c in coverage.items()},
            "ramp_threshold_mwh": threshold,
            "ramp_threshold_method": f"{RAMP_PERCENTILE:g}th percentile (numpy linear) of |y_t - y_(t-1)| over "
                                     f"{n_ramp_pairs} consecutive valid training pairs",
            "daylight_proxy": f"aggregated target-hour shortwave_radiation (GHI) > {DAYLIGHT_GHI_THRESHOLD:g} W/m^2; "
                              "retrospective grouping only, never a model input or selection criterion",
            "trace_block": trace_block(windows, table),
            "scaler": {"fit_rows": scaler["n_fit_rows"], "fit_row_index_sha256": scaler["fit_row_index_sha256"],
                       "zero_variance_columns": [c for c, v in scaler["columns"].items() if v["zero_variance"]]},
            "quality_report": quality,
            "sources": {
                "collection_record": collection_file.name,
                "collected_at_utc": collection["collected_at_utc"],
                "attribution": collection["attribution"],
                "documentation": collection["documentation"],
                "ieso_files": [{k: r.get(k) for k in ("filename", "url", "revision", "sha256", "etag", "last_modified",
                                                       "retrieved_at_utc", "doc_header", "snapshot")}
                               for r in collection["ieso"]["files"]],
                "open_meteo_requests": {loc: [{k: r.get(k) for k in ("params", "sha256", "retrieved_at_utc",
                                                                    "response_metadata", "snapshot")}
                                              for r in recs] for loc, recs in collection["open_meteo"].items()},
                "ieso_unit": ieso.UNIT,
                "ieso_scope": "registered Ontario generators >= 20 MW (not all Ontario or rooftop solar)",
                "weather_units": open_meteo.EXPECTED_UNITS,
                "weather_aggregation_in_time": open_meteo.AGGREGATION,
                "weather_model": open_meteo.MODEL,
                "weather_spatial_proxy": "equal-weight mean over the configured proxy locations (not plant "
                                         "locations or capacity weights)",
            },
            "files": files,
            "precision": "float32 tensors; float64 metrics",
        }
        atomic_write_json(tmp / "dataset_manifest.json", manifest)
        (tmp / "quality_report.json").write_text(json.dumps(quality, indent=2, sort_keys=True, default=str),
                                                  encoding="utf-8")
        final = data_dir / "processed" / dataset_id
        if final.exists():
            verify_processed(final)  # identical content identity: reuse, never overwrite
            log(f"dataset {dataset_id} already exists; reusing {final}")
        else:
            os.replace(tmp, final)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)

    pointer = data_dir / f"prepared_{scale_label}.json"
    atomic_write_json(pointer, {"dataset_id": dataset_id, "scale_label": scale_label,
                                "processed_dir": os.path.relpath(final, pointer.parent),
                                "manifest": os.path.relpath(final / "dataset_manifest.json", pointer.parent)})
    log(f"dataset {dataset_id}: {manifest['window_counts']}; pointer {pointer}")
    return manifest, pointer


def _ensure(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def verify_processed(processed_dir):
    """Re-hash the processed files against the manifest; returns the manifest."""
    processed_dir = Path(processed_dir)
    manifest = json.loads((processed_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["files"].items():
        if sha256_file(processed_dir / name) != digest:
            raise RuntimeError(f"{processed_dir / name} does not match the dataset manifest checksum")
    return manifest


# ---------------------------------------------------------------------------------------------
# Loading a prepared dataset
# ---------------------------------------------------------------------------------------------

class PreparedDataset:
    """A verified processed dataset: hourly table, windows, scaler, and torch datasets per split."""

    def __init__(self, path):
        path = Path(path)
        if path.name.startswith("prepared_"):
            pointer = json.loads(path.read_text(encoding="utf-8"))
            path = (path.parent / pointer["processed_dir"]).resolve()
        elif path.name == "dataset_manifest.json":
            path = path.parent
        self.dir = path
        self.manifest = verify_processed(path)
        self.dataset_id = self.manifest["dataset_id"]
        self.sequence_length = self.manifest["sequence_length"]
        self.scaler = json.loads((path / "scaler.json").read_text(encoding="utf-8"))
        self.table = pd.read_csv(path / "hourly_table.csv", keep_default_na=True)
        self.table["invalid_reasons"] = self.table["invalid_reasons"].fillna("")
        self.windows = pd.read_csv(path / "windows.csv")
        for col in ("is_daylight_proxy",):
            self.windows[col] = self.windows[col].astype("boolean")
        if list(self.scaler["feature_order"]) != list(FEATURE_COLUMNS):
            raise ValueError("scaler feature order does not match FEATURE_COLUMNS")
        self.features = apply_scaler(self.table, self.scaler)
        self.targets_scaled = scale_target(self.table[TARGET_COLUMN], self.scaler).astype(np.float32)

    def split_windows(self, split):
        w = self.windows
        return w[(w["split"] == split) & w["eligible"]].reset_index(drop=True)

    def torch_dataset(self, split):
        return SolarWindowDataset(self.features, self.targets_scaled, self.split_windows(split),
                                  self.sequence_length, split)
