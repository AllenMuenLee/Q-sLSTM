"""Hourly table, features, windows, chronological splits, scaler, and dataset identity."""

import datetime as dt
import json
import math

import numpy as np
import pandas as pd
import pytest

from solar_test_utils import build_synthetic_dataset, synthetic_solar
from q_slstm.datasets import solar_generation as ds
from q_slstm.datasets.solar_generation import FEATURE_COLUMNS, PreparedDataset, calendar_features, expected_grid


@pytest.fixture(scope="module")
def data(synthetic_pointer):
    return PreparedDataset(synthetic_pointer)


# ---------------------------------------------------------------------------------------------
# 5. Features and metadata
# ---------------------------------------------------------------------------------------------

def test_exactly_thirteen_ordered_features():
    assert FEATURE_COLUMNS == (
        "solar_generation", "shortwave_radiation", "direct_radiation", "diffuse_radiation", "cloud_cover",
        "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation",
        "hour_sin", "hour_cos", "day_sin", "day_cos")
    assert ds.RAW_INPUT_SIZE == 13 and len(ds.PHYSICAL_COLUMNS) == 9


def test_cyclic_calendar_values_use_interval_start_in_fixed_est():
    grid = expected_grid(dt.date(2024, 1, 1), dt.date(2024, 1, 1))
    cal = calendar_features(grid["interval_start_utc"])
    # hour ending 1 starts at 00:00 EST on day 1: hour label 0, day index 0
    assert cal.loc[0].tolist() == pytest.approx([0.0, 1.0, 0.0, 1.0])
    # hour ending 7 starts at 06:00 EST: angle 2*pi*6/24 = pi/2
    assert cal.loc[6, ["hour_sin", "hour_cos"]].tolist() == pytest.approx([1.0, 0.0], abs=1e-12)
    leap = calendar_features(pd.DatetimeIndex([pd.Timestamp("2024-12-31T05:00Z")]))  # 00:00 EST Dec 31
    assert leap["day_sin"].iloc[0] == pytest.approx(math.sin(2 * math.pi * 365 / 366))
    common = calendar_features(pd.DatetimeIndex([pd.Timestamp("2025-12-31T05:00Z")]))
    assert common["day_sin"].iloc[0] == pytest.approx(math.sin(2 * math.pi * 364 / 365))
    july = calendar_features(pd.DatetimeIndex([pd.Timestamp("2024-07-01T17:00Z")]))  # 12:00 EST, no DST
    assert july["hour_cos"].iloc[0] == pytest.approx(-1.0)


def test_items_contain_only_declared_inputs_and_past_timestamps(data):
    item = data.torch_dataset("val")[0]
    L = data.sequence_length
    assert tuple(item["inputs"].shape) == (L, 13) and tuple(item["target"].shape) == (1,)
    assert set(item) == {"inputs", "target", "window_id", "origin_timestamp_utc", "target_timestamp_utc", "split"}
    j = item["window_id"]
    rows = data.table.iloc[j - L:j]
    expected = ds.apply_scaler(rows, data.scaler)
    np.testing.assert_allclose(item["inputs"].numpy(), expected)
    input_times = pd.to_datetime(rows["timestamp_utc"])
    target_time = pd.Timestamp(item["target_timestamp_utc"])
    assert (input_times < target_time).all()
    assert input_times.iloc[-1] == pd.Timestamp(item["origin_timestamp_utc"])
    assert target_time - input_times.iloc[-1] == pd.Timedelta(hours=1)


def test_generation_and_radiation_join_on_the_same_interval(data):
    t = data.table.set_index("timestamp_utc")
    # IESO 2024-06-03 hour ending 13 = interval (17:00, 18:00] UTC; synthetic weather stamped 18:00 UTC
    # describes the same interval, so GHI = 3 * generation + (lat - 43) averaged over the two locations.
    row = t.loc["2024-06-03T18:00:00Z"]
    assert (row["source_date"], row["source_hour"]) == ("2024-06-03", 13)
    assert row["solar_generation"] == synthetic_solar(dt.date(2024, 6, 3), 13)
    assert row["shortwave_radiation"] == pytest.approx(3 * row["solar_generation"] + 0.5)


# ---------------------------------------------------------------------------------------------
# 6. Splits and windows
# ---------------------------------------------------------------------------------------------

def test_chronological_split_boundaries_and_target_assignment(data):
    n = data.manifest["n_rows"]
    assert n == 6 * 24
    train_end, val_end = ds.split_boundaries(n)
    assert (train_end, val_end) == (100, 115)
    assert ds.split_boundaries(17544) == (12280, 14035)
    b = data.manifest["split_boundaries"]
    assert b["train"]["rows"] == [0, 100] and b["test"]["rows"] == [115, 144]
    w = data.windows
    assert (w.loc[w["split"] == "train", "target_index"] < 100).all()
    assert w.loc[w["split"] == "val", "target_index"].between(100, 114).all()
    assert (w.loc[w["split"] == "test", "target_index"] >= 115).all()
    ids = [set(data.split_windows(s)["window_id"]) for s in ("train", "val", "test")]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    first_val = data.split_windows("val").iloc[0]
    assert first_val["origin_index"] == 99  # historical inputs may cross the boundary; labels never do
    assert w["window_id"].min() == data.sequence_length


def test_gaps_and_flags_invalidate_windows_instead_of_compressing_time(tmp_path):
    missing = {(dt.date(2024, 6, 2), 5)}
    flags = {(dt.date(2024, 6, 2), 9): -1}
    pointer = build_synthetic_dataset(tmp_path, flags=flags, missing=missing, min_coverage=0.9)
    data = PreparedDataset(pointer)
    t = data.table
    gap = t.index[(t["source_date"] == "2024-06-02") & (t["source_hour"] == 5)][0]
    flag = t.index[(t["source_date"] == "2024-06-02") & (t["source_hour"] == 9)][0]
    assert not t.loc[gap, "row_valid"] and "generation_missing" in t.loc[gap, "invalid_reasons"]
    assert not t.loc[flag, "row_valid"] and "generation_quality_flag_-1" in t.loc[flag, "invalid_reasons"]
    assert np.isnan(t.loc[gap, "solar_generation"])  # missing stays missing, never zero
    assert len(t) == 6 * 24  # the grid is complete; nothing was deleted
    L = data.sequence_length
    w = data.windows.set_index("target_index")
    for bad in (gap, flag):
        touching = range(bad, bad + L + 1)  # windows whose inputs or target include the bad row
        assert not w.loc[list(touching), "eligible"].any()
    assert w.loc[flag + L + 1, "eligible"] and w.loc[gap - 1, "eligible"]  # untouched neighbors stay usable
    assert pd.read_csv(data.dir / "quarantine.csv")["source_hour"].tolist() == [5, 9]
    q = data.manifest["quality_report"]
    assert q["generation_status_counts"] == {"ok": 142, "missing": 1, "flagged": 1}


def test_coverage_threshold_fails_preparation(tmp_path):
    flags = {(dt.date(2024, 6, 6), h): -1 for h in range(1, 10)}  # 9 of 29 test rows flagged
    with pytest.raises(ds.DatasetQualityError, match="test: valid rows"):
        build_synthetic_dataset(tmp_path, flags=flags)
    report = json.loads((tmp_path / "quality_report_pilot-overridden_FAILED.json").read_text())
    assert report["coverage_by_split"]["test"]["valid_rows"] == 20


def test_missing_weather_location_invalidates_rows(tmp_path):
    def drop_west_value(hourly, params):
        if params["latitude"].startswith("43.") and params["start_date"] == "2024-06-01":
            hourly["temperature_2m"][30] = None  # 2024-06-02T06:00Z at the west location

    data = PreparedDataset(build_synthetic_dataset(tmp_path, weather_override=drop_west_value))
    row = data.table.set_index("timestamp_utc").loc["2024-06-02T06:00:00Z"]
    assert not row["row_valid"] and row["invalid_reasons"] == "west:temperature_2m_missing"
    assert np.isnan(row["temperature_2m"])


# ---------------------------------------------------------------------------------------------
# 7. Scaler
# ---------------------------------------------------------------------------------------------

def test_scaler_uses_training_rows_only_and_inverts(data):
    table = data.table.copy()
    scaler = ds.fit_scaler(table)
    held_out = table["row_split"] != "train"
    table.loc[held_out, list(ds.PHYSICAL_COLUMNS)] = 1e6  # arbitrary held-out changes
    assert ds.fit_scaler(table) == scaler
    train = data.table[(data.table["row_split"] == "train") & data.table["row_valid"]]
    assert scaler["target"]["mean"] == pytest.approx(train["solar_generation"].mean())
    assert scaler["target"]["scale"] == pytest.approx(train["solar_generation"].std(ddof=0))
    assert scaler["columns"]["hour_sin"]["scale"] == 1.0 and scaler["columns"]["hour_sin"]["mean"] == 0.0
    y = np.array([0.0, 12.5, 300.0])
    np.testing.assert_allclose(ds.unscale_target(ds.scale_target(y, scaler), scaler), y)
    scaled = ds.apply_scaler(table, scaler)
    assert np.nanmax(scaled[held_out.to_numpy(), 0]) > 100  # held-out values may exceed the training range


def test_constant_feature_uses_unit_scale(data):
    table = data.table.copy()
    table["precipitation"] = 0.0
    c = ds.fit_scaler(table)["columns"]["precipitation"]
    assert c["zero_variance"] and c["scale"] == 1.0 and c["mean"] == 0.0


def test_dataset_identity_is_content_addressed_and_verified(tmp_path, synthetic_pointer, data):
    again = PreparedDataset(build_synthetic_dataset(tmp_path))
    assert again.dataset_id == data.dataset_id  # same sources and settings, different paths and times
    manifest = data.manifest
    assert manifest["raw_input_size"] == 13 and manifest["feature_columns"] == list(FEATURE_COLUMNS)
    assert manifest["identity"]["locations"][0]["weight"] == 0.5 and manifest["sources"]["ieso_unit"] == "MWh"
    assert manifest["quality_report"]["alignment_diagnostic"]["apr_oct"]["best_lag"] == 0
    other = PreparedDataset(build_synthetic_dataset(tmp_path / "L6", sequence_length=6))
    assert other.dataset_id != data.dataset_id
    (again.dir / "windows.csv").write_text("tampered", encoding="utf-8")
    with pytest.raises(RuntimeError, match="checksum"):
        PreparedDataset(again.dir)
