# Open-Meteo historical weather (ERA5 reanalysis) adapter.
#
# API: https://archive-api.open-meteo.com/v1/archive
# Docs: https://open-meteo.com/en/docs/historical-weather-api
# Variable semantics (https://open-meteo.com/en/docs/historical-weather-api#hourly-parameter-definition):
#   shortwave/direct/diffuse radiation: mean over the preceding hour (direct_radiation is on the
#   horizontal plane, not DNI); precipitation: sum over the preceding hour; temperature, humidity,
#   cloud cover, wind speed: instantaneous at the timestamp. With timezone=UTC, the value stamped T
#   therefore describes the completed interval (T - 1h, T] and joins the IESO interval ending at T.
#
# The archive response does not echo the model; the requested `models=era5` is recorded instead.

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pandas as pd

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
DOCS_URL = "https://open-meteo.com/en/docs/historical-weather-api"
ATTRIBUTION = "Weather data by Open-Meteo.com (CC BY 4.0), ERA5 reanalysis (Copernicus Climate Change Service)"
MODEL = "era5"

WEATHER_VARIABLES = (
    "shortwave_radiation", "direct_radiation", "diffuse_radiation", "cloud_cover",
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation",
)
EXPECTED_UNITS = {
    "time": "iso8601",
    "shortwave_radiation": "W/m²", "direct_radiation": "W/m²", "diffuse_radiation": "W/m²",
    "cloud_cover": "%", "temperature_2m": "°C", "relative_humidity_2m": "%",
    "wind_speed_10m": "km/h", "precipitation": "mm",
}
AGGREGATION = {
    "shortwave_radiation": "preceding-hour mean", "direct_radiation": "preceding-hour mean",
    "diffuse_radiation": "preceding-hour mean", "precipitation": "preceding-hour sum",
    "cloud_cover": "instantaneous", "temperature_2m": "instantaneous",
    "relative_humidity_2m": "instantaneous", "wind_speed_10m": "instantaneous",
}
# Physical-range checks; violations are quarantined with a reason, never clipped.
NONNEGATIVE = ("shortwave_radiation", "direct_radiation", "diffuse_radiation", "wind_speed_10m", "precipitation")
PERCENT = ("cloud_cover", "relative_humidity_2m")
MAX_GRID_DISTANCE_DEG = 0.5  # resolved ERA5 cell (0.25 deg grid) must lie near the requested point


class OpenMeteoSchemaError(ValueError):
    """The archive response does not match the requested variables, units, or time axis."""


def request_params(latitude, longitude, start_date, end_date):
    """Complete query parameters for one archive request (dates are inclusive UTC days)."""
    return {
        "latitude": f"{latitude:.4f}",
        "longitude": f"{longitude:.4f}",
        "start_date": str(start_date),
        "end_date": str(end_date),
        "hourly": ",".join(WEATHER_VARIABLES),
        "timezone": "UTC",
        "models": MODEL,
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
    }


def required_utc_range(first_source_date, last_source_date):
    """UTC dates covering every IESO interval end of the EST source dates (end of last day + 1)."""
    return first_source_date, last_source_date + dt.timedelta(days=1)


def monthly_chunks(start_utc_date, end_utc_date):
    """Monthly request windows, each padded one day into the next month, clipped to the range.

    Consecutive chunks therefore share their boundary day, which `combine_chunks` deduplicates.
    """
    chunks = []
    month = dt.date(start_utc_date.year, start_utc_date.month, 1)
    while month <= end_utc_date:
        nxt = dt.date(month.year + (month.month == 12), month.month % 12 + 1, 1)
        lo, hi = max(month, start_utc_date), min(nxt, end_utc_date)
        chunks.append((f"{month:%Y-%m}", lo, hi))
        month = nxt
    return chunks


def parse_archive_json(payload, params, source=""):
    """Validate one archive response and return (metadata, DataFrame[timestamp_utc, 8 variables])."""
    where = source or f"{params['latitude']},{params['longitude']} {params['start_date']}..{params['end_date']}"
    if not isinstance(payload, dict):
        raise OpenMeteoSchemaError(f"{where}: response is not a JSON object")
    if payload.get("error"):
        raise OpenMeteoSchemaError(f"{where}: API error: {payload.get('reason')}")
    for key in ("hourly", "hourly_units", "latitude", "longitude", "utc_offset_seconds"):
        if key not in payload:
            raise OpenMeteoSchemaError(f"{where}: missing field {key!r}")
    if payload["utc_offset_seconds"] != 0:
        raise OpenMeteoSchemaError(f"{where}: utc_offset_seconds={payload['utc_offset_seconds']}, expected 0")
    units, hourly = payload["hourly_units"], payload["hourly"]
    for name, unit in EXPECTED_UNITS.items():
        if units.get(name) != unit:
            raise OpenMeteoSchemaError(f"{where}: unit of {name} is {units.get(name)!r}, expected {unit!r}")
        if name not in hourly:
            raise OpenMeteoSchemaError(f"{where}: hourly variable {name!r} missing")
    extra = set(hourly) - set(EXPECTED_UNITS)
    if extra:
        raise OpenMeteoSchemaError(f"{where}: unexpected hourly variables {sorted(extra)}")

    times = hourly["time"]
    start = dt.date.fromisoformat(params["start_date"])
    end = dt.date.fromisoformat(params["end_date"])
    n_expected = ((end - start).days + 1) * 24
    lengths = {name: len(hourly[name]) for name in EXPECTED_UNITS}
    if set(lengths.values()) != {n_expected}:
        raise OpenMeteoSchemaError(f"{where}: array lengths {lengths}, expected {n_expected}")
    stamps = pd.to_datetime(pd.Series(times), format="%Y-%m-%dT%H:%M", utc=True)
    expected = pd.date_range(pd.Timestamp(start, tz="UTC"), periods=n_expected, freq="h")
    if not (pd.DatetimeIndex(stamps) == expected).all():
        raise OpenMeteoSchemaError(f"{where}: time axis is not the contiguous hourly UTC range {start}..{end}")

    req_lat, req_lon = float(params["latitude"]), float(params["longitude"])
    lat, lon = float(payload["latitude"]), float(payload["longitude"])
    if abs(lat - req_lat) > MAX_GRID_DISTANCE_DEG or abs(lon - req_lon) > MAX_GRID_DISTANCE_DEG:
        raise OpenMeteoSchemaError(f"{where}: resolved grid point ({lat}, {lon}) far from request ({req_lat}, {req_lon})")

    frame = pd.DataFrame({"timestamp_utc": expected})
    for name in WEATHER_VARIABLES:
        values = [math.nan if v is None else v for v in hourly[name]]
        try:
            frame[name] = np.asarray(values, dtype=float)
        except (TypeError, ValueError):
            raise OpenMeteoSchemaError(f"{where}: non-numeric values in {name}") from None
    meta = {
        "requested_latitude": req_lat, "requested_longitude": req_lon,
        "resolved_latitude": lat, "resolved_longitude": lon,
        "elevation_m": payload.get("elevation"), "timezone": payload.get("timezone"),
        "requested_model": params.get("models"), "model_echoed_by_api": payload.get("model"),
        "units": {k: units[k] for k in EXPECTED_UNITS},
    }
    return meta, frame


def combine_chunks(frames):
    """Concatenate monthly frames; identical boundary duplicates collapse, conflicting ones fail.

    Returns (frame sorted by timestamp, number of collapsed duplicate rows).
    """
    combined = pd.concat(frames, ignore_index=True).sort_values("timestamp_utc", kind="stable")
    dup_mask = combined.duplicated("timestamp_utc", keep=False)
    collapsed = 0
    if dup_mask.any():
        for ts, group in combined[dup_mask].groupby("timestamp_utc"):
            values = group[list(WEATHER_VARIABLES)].to_numpy(dtype=float)
            same = np.all((values == values[0]) | (np.isnan(values) & np.isnan(values[0])))
            if not same:
                raise OpenMeteoSchemaError(f"conflicting duplicate weather values at {ts}")
            collapsed += len(group) - 1
        combined = combined.drop_duplicates("timestamp_utc", keep="first")
    return combined.reset_index(drop=True), collapsed


def validate_weather(frame):
    """Per-row list of range-violation reasons for one location (empty list = valid or missing)."""
    reasons = [[] for _ in range(len(frame))]
    for name in WEATHER_VARIABLES:
        values = frame[name].to_numpy(dtype=float)
        finite = np.isfinite(values)
        bad = np.zeros(len(values), dtype=bool)
        if name in NONNEGATIVE:
            bad |= finite & (values < 0)
        if name in PERCENT:
            bad |= finite & ((values < 0) | (values > 100))
        bad |= np.isinf(values)
        for i in np.flatnonzero(bad):
            reasons[i].append(f"{name}_out_of_range")
    return reasons


def aggregate_locations(per_location, weights, grid):
    """Weighted mean of each variable over locations at each grid timestamp.

    `per_location`: {location_id: frame with timestamp_utc + variables}; `weights`: {location_id: w}
    summing to 1. A timestamp missing or invalid at any location yields NaN for that variable plus a
    reason; weights are never renormalized over the remaining locations.
    Returns (aggregate frame on `grid`, list of reason lists per grid row).
    """
    if set(per_location) != set(weights):
        raise ValueError(f"locations with data {sorted(per_location)} != configured {sorted(weights)}")
    total = sum(weights.values())
    if not math.isclose(total, 1.0, abs_tol=1e-9):
        raise ValueError(f"location weights must sum to 1, got {total}")
    grid = pd.DatetimeIndex(grid)
    out = pd.DataFrame({"timestamp_utc": grid})
    reasons = [[] for _ in range(len(grid))]
    acc = {name: np.zeros(len(grid)) for name in WEATHER_VARIABLES}
    for loc in sorted(weights):
        frame = per_location[loc].set_index("timestamp_utc").reindex(grid)
        loc_reasons = validate_weather(frame.reset_index())
        for name in WEATHER_VARIABLES:
            values = frame[name].to_numpy(dtype=float)
            acc[name] += weights[loc] * values  # NaN propagates: no renormalization
            for i in np.flatnonzero(np.isnan(values)):
                reasons[i].append(f"{loc}:{name}_missing")
        for i, r in enumerate(loc_reasons):
            reasons[i] += [f"{loc}:{x}" for x in r]
    for name in WEATHER_VARIABLES:
        out[name] = acc[name]
    return out, reasons


def read_locations(path):
    """Location proxy table: location_id, latitude, longitude, weight (weights must sum to 1)."""
    table = pd.read_csv(path)
    required = ["location_id", "latitude", "longitude", "weight"]
    if list(table.columns) != required:
        raise ValueError(f"{path}: columns must be {required}, got {list(table.columns)}")
    if table["location_id"].duplicated().any():
        raise ValueError(f"{path}: duplicate location_id")
    if not math.isclose(table["weight"].sum(), 1.0, abs_tol=1e-9) or (table["weight"] <= 0).any():
        raise ValueError(f"{path}: weights must be positive and sum to 1")
    return table
