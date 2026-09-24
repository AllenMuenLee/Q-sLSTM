"""Shared offline helpers and fixtures (registered via tests/conftest.py) for the solar-generation tests: fixtures, a fake HTTP transport, and a
synthetic IESO / Open-Meteo source that exercises the real collector and preparation code."""

import datetime as dt
import json
import math
import urllib.parse
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = REPO_ROOT / "scripts" / "experiments" / "solar_generation"
LOCATIONS_FILE = REPO_ROOT / "configs" / "solar_generation" / "locations.csv"

from q_slstm.data_sources import ieso, open_meteo
from q_slstm.data_sources.http import HttpClient, RetryPolicy

FIXTURES = REPO_ROOT / "tests" / "fixtures" / "solar_generation"
LOCATIONS = pd.DataFrame({"location_id": ["east", "west"], "latitude": [44.0, 43.0],
                          "longitude": [-77.0, -80.0], "weight": [0.5, 0.5]})


@pytest.fixture
def fixture_dir():
    return FIXTURES


# ---------------------------------------------------------------------------------------------
# Synthetic sources
# ---------------------------------------------------------------------------------------------

def synthetic_solar(day, hour_ending):
    """Deterministic SOLAR MWh for an EST date / hour ending: zero at night, a daily bump by day."""
    start_hour = hour_ending - 1
    shape = math.sin(math.pi * (start_hour - 5) / 14) if 5 <= start_hour <= 19 else 0.0
    return int(round(max(0.0, shape) * (250 + 10 * (day.toordinal() % 7))))


def synthetic_ieso_xml(year, days, flags=None, missing=None, fuel_first="WIND"):
    """Annual report XML for `days` (dates in `year`); `flags` / `missing` are {(date, hour): quality}."""
    flags, missing = flags or {}, missing or set()
    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           f'<Document docID="GenOutputbyFuelHourly" xmlns="{ieso.NS}">',
           "<DocHeader><DocTitle>Generator Output by Fuel Type Hourly Report</DocTitle><DocRevision>1</DocRevision>"
           "<CreatedAt>2025-01-01T06:50:47</CreatedAt></DocHeader>",
           f"<DocBody><DeliveryYear>{year}</DeliveryYear>"]
    for day in days:
        out.append(f"<DailyData><Day>{day.isoformat()}</Day>")
        for h in range(1, 25):
            wind = f"<FuelTotal><Fuel>WIND</Fuel><EnergyValue><OutputQuality>0</OutputQuality><Output>{500 + h}</Output></EnergyValue></FuelTotal>"
            if (day, h) in missing:
                solar = "<FuelTotal><Fuel>SOLAR</Fuel><EnergyValue><OutputQuality>0</OutputQuality></EnergyValue></FuelTotal>"
            else:
                q = flags.get((day, h), 0)
                solar = (f"<FuelTotal><Fuel>SOLAR</Fuel><EnergyValue><OutputQuality>{q}</OutputQuality>"
                         f"<Output>{synthetic_solar(day, h)}</Output></EnergyValue></FuelTotal>")
            fuels = wind + solar if fuel_first == "WIND" else solar + wind
            out.append(f"<HourlyData><Hour>{h}</Hour>{fuels}</HourlyData>")
        out.append("</DailyData>")
    out.append("</DocBody></Document>")
    return "\n".join(out).encode("utf-8")


def synthetic_weather_payload(params, override=None):
    """Open-Meteo-shaped JSON for a request: GHI tracks the synthetic generation interval by interval."""
    start = dt.date.fromisoformat(params["start_date"])
    end = dt.date.fromisoformat(params["end_date"])
    times = pd.date_range(pd.Timestamp(start), periods=((end - start).days + 1) * 24, freq="h")
    lat = float(params["latitude"])
    hourly = {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in times]}
    ghi = []
    for t in times:
        est_start = (t - pd.Timedelta(hours=6)).to_pydatetime()  # interval (t-1h, t] starts at t-6h EST
        hour_ending = est_start.hour + 1
        ghi.append(3.0 * synthetic_solar(est_start.date(), hour_ending) + (lat - 43.0))
    hourly["shortwave_radiation"] = [max(0.0, g) for g in ghi]
    hourly["direct_radiation"] = [0.6 * g for g in hourly["shortwave_radiation"]]
    hourly["diffuse_radiation"] = [0.4 * g for g in hourly["shortwave_radiation"]]
    hours = [int(t.timestamp() // 3600) for t in times]  # values depend on the timestamp only
    hourly["cloud_cover"] = [float((k * 7) % 100) for k in hours]
    hourly["temperature_2m"] = [10.0 + lat / 10 + 5 * math.sin(k / 5) for k in hours]
    hourly["relative_humidity_2m"] = [60.0 + (k % 30) for k in hours]
    hourly["wind_speed_10m"] = [5.0 + (k % 11) for k in hours]
    hourly["precipitation"] = [0.1 * (k % 3 == 0) for k in hours]
    if override:
        override(hourly, params)
    return {"latitude": round(lat * 4) / 4, "longitude": round(float(params["longitude"]) * 4) / 4,
            "generationtime_ms": 0.1, "utc_offset_seconds": 0, "timezone": "GMT", "timezone_abbreviation": "GMT",
            "elevation": 100.0, "hourly_units": dict(open_meteo.EXPECTED_UNITS), "hourly": hourly}


class FakeServer:
    """Transport serving the IESO index, annual XML files (with Range support), and archive JSON."""

    def __init__(self, xml_by_year, weather_override=None, last_modified="Wed, 01 Jan 2025 11:51:16 GMT"):
        self.xml = {f"PUB_GenOutputbyFuelHourly_{y}.xml": b for y, b in xml_by_year.items()}
        self.weather_override = weather_override
        self.last_modified = last_modified
        self.calls = []

    def index(self):
        links = "".join(f'<a href="{n}">{n}</a>\n' for n in [*self.xml, "PUB_GenOutputbyFuelHourly.xml"])
        return f"<html><body><pre>{links}</pre></body></html>".encode()

    def __call__(self, url, headers, connect_timeout, read_timeout):
        self.calls.append((url, dict(headers)))
        parts = urllib.parse.urlsplit(url)
        if url.startswith(open_meteo.ARCHIVE_URL):
            params = dict(urllib.parse.parse_qsl(parts.query))
            body = json.dumps(synthetic_weather_payload(params, self.weather_override)).encode()
            return 200, {"content-length": str(len(body)), "content-type": "application/json"}, body
        name = parts.path.rsplit("/", 1)[-1]
        if name == "":
            body = self.index()
            return 200, {"content-length": str(len(body))}, body
        if name not in self.xml:
            return 404, {}, b"not found"
        data = self.xml[name]
        rng = headers.get("Range")
        if rng:
            lo, hi = map(int, rng.split("=")[1].split("-"))
            hi = min(hi, len(data) - 1)
            chunk = data[lo:hi + 1]
            return 206, {"content-range": f"bytes {lo}-{hi}/{len(data)}", "content-length": str(len(chunk)),
                         "last-modified": self.last_modified, "etag": '"abc"'}, chunk
        return 200, {"content-length": str(len(data)), "last-modified": self.last_modified}, data


def fake_client(server):
    return HttpClient(RetryPolicy(max_attempts=5, backoff_base=0.0, jitter=0.0), transport=server,
                      sleep=lambda s: None)


def build_synthetic_dataset(data_dir, start="2024-06-01", end="2024-06-06", sequence_length=4, flags=None,
                            missing=None, weather_override=None, min_coverage=0.95):
    """Run the real collector (fake transport) and preparation on synthetic sources; returns pointer path."""
    from q_slstm.data_sources.collect import collect
    from q_slstm.datasets.solar_generation import prepare_dataset

    data_dir = Path(data_dir)
    days = pd.date_range(start, end).date
    server = FakeServer({int(start[:4]): synthetic_ieso_xml(int(start[:4]), days, flags, missing)}, weather_override)
    loc_file = data_dir / "locations.csv"
    data_dir.mkdir(parents=True, exist_ok=True)
    LOCATIONS.to_csv(loc_file, index=False)
    _, collection = collect("pilot", data_dir, loc_file, start, end, client=fake_client(server), log=lambda *a: None)
    _, pointer = prepare_dataset(data_dir, "pilot", collection, sequence_length, min_coverage, log=lambda *a: None)
    return pointer


@pytest.fixture(scope="session")
def synthetic_pointer(tmp_path_factory):
    return build_synthetic_dataset(tmp_path_factory.mktemp("solar_data"))
