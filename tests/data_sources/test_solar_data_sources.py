"""IESO / Open-Meteo adapters, HTTP retries, snapshot cache, and the interval-time convention."""

import datetime as dt
import json
import math
import os

import numpy as np
import pandas as pd
import pytest

from solar_test_utils import FakeServer, fake_client, synthetic_ieso_xml, synthetic_weather_payload
from q_slstm.data_sources import cache, ieso, open_meteo
from q_slstm.data_sources.collect import collect, resolve_dates
from q_slstm.data_sources.http import HttpClient, HttpError, RetryPolicy, TransientHttpError, parse_retry_after

NS = ieso.NS


def _fixture_frame(fixture_dir):
    return ieso.parse_generation_xml((fixture_dir / "ieso_gen_output_2024_excerpt.xml").read_bytes(), "fixture")


# ---------------------------------------------------------------------------------------------
# 1. IESO parsing
# ---------------------------------------------------------------------------------------------

def test_real_fixture_extracts_solar_with_quality_flags(fixture_dir):
    header, frame = _fixture_frame(fixture_dir)
    assert header["delivery_year"] == 2024 and header["doc_revision"] == "1"
    row = lambda d, h: frame[(frame["source_date"] == d) & (frame["source_hour"] == h)].iloc[0]
    # Values copied from the published 2024 file.
    assert row("2024-04-13", 9)["solar_generation"] == 94.0
    assert row("2024-04-13", 9)["output_quality"] == -1 and row("2024-04-13", 9)["status"] == "flagged"
    assert row("2024-04-13", 10)["solar_generation"] == 141.0 and row("2024-04-13", 10)["status"] == "flagged"
    assert row("2024-03-10", 1)["solar_generation"] == 0.0 and row("2024-03-10", 1)["status"] == "ok"
    assert set(frame["source_hour"][frame["source_date"] == "2024-12-31"]) == {23, 24}
    assert frame["solar_generation"].dtype == float and (frame["solar_generation"] >= 0).all()


def _parse(xml):
    return ieso.parse_generation_xml(xml.encode("utf-8") if isinstance(xml, str) else xml, "test.xml")


def _doc(hours_xml, day="2024-06-01"):
    return (f'<Document docID="GenOutputbyFuelHourly" xmlns="{NS}"><DocBody><DeliveryYear>2024</DeliveryYear>'
            f"<DailyData><Day>{day}</Day>{hours_xml}</DailyData></DocBody></Document>")


def _hour(h, *fuels):
    return f"<HourlyData><Hour>{h}</Hour>{''.join(fuels)}</HourlyData>"


def _fuel(name, output="12", quality="0"):
    out = "" if output is None else f"<Output>{output}</Output>"
    return f"<FuelTotal><Fuel>{name}</Fuel><EnergyValue><OutputQuality>{quality}</OutputQuality>{out}</EnergyValue></FuelTotal>"


def test_extraction_is_independent_of_fuel_order_and_uses_only_solar():
    day = dt.date(2024, 6, 1)
    a = _parse(synthetic_ieso_xml(2024, [day], fuel_first="WIND"))[1]
    b = _parse(synthetic_ieso_xml(2024, [day], fuel_first="SOLAR"))[1]
    pd.testing.assert_frame_equal(a, b)
    one = _parse(_doc(_hour(1, _fuel("WIND", "900"), _fuel("SOLAR", "7"), _fuel("NUCLEAR", "9000"))))[1]
    assert one["solar_generation"].tolist() == [7.0]


def test_missing_fields_are_missing_observations_not_zero():
    xml = _doc(_hour(1, _fuel("SOLAR", None)) + _hour(2, "<FuelTotal><Fuel>SOLAR</Fuel></FuelTotal>")
               + _hour(3, _fuel("WIND")))
    frame = _parse(xml)[1].set_index("source_hour")
    assert frame["solar_generation"].isna().all()
    assert (frame["status"] == "missing").all()
    assert frame.loc[1, "energy_value_present"] and not frame.loc[1, "output_present"]
    assert not frame.loc[2, "energy_value_present"]
    assert not frame.loc[3, "fuel_record_present"]


@pytest.mark.parametrize("bad, message", [
    ("<Document", "malformed"),
    (f'<Document docID="Other" xmlns="{NS}"><DocBody/></Document>', "docID"),
    ('<Document docID="GenOutputbyFuelHourly"><DocBody/></Document>', "root element"),
    (_doc(_hour(1, _fuel("SOLAR").replace("</EnergyValue>", "<OutputMW>3</OutputMW></EnergyValue>"))), "OutputMW"),
    (_doc(_hour(25, _fuel("SOLAR"))), "outside 1..24"),
    (_doc(_hour(1, _fuel("SOLAR", "abc"))), "non-numeric"),
    (_doc(_hour(1, _fuel("SOLAR"), _fuel("SOLAR"))), "SOLAR records"),
    (_doc(_hour(1, _fuel("SOLAR", "5")) + _hour(1, _fuel("SOLAR", "6"))), "conflicting duplicate"),
    (_doc(_hour(1, _fuel("SOLAR")), day="2023-12-31"), "outside DeliveryYear"),
])
def test_malformed_payloads_and_schema_changes_fail_with_location(bad, message):
    with pytest.raises(ieso.IesoSchemaError, match=message) as err:
        _parse(bad)
    assert "test.xml" in str(err.value)


def test_identical_duplicate_hours_collapse_with_audit_count():
    header, frame = _parse(_doc(_hour(1, _fuel("SOLAR", "5")) + _hour(1, _fuel("SOLAR", "5"))))
    assert len(frame) == 1 and header["duplicates_collapsed"] == 1


# ---------------------------------------------------------------------------------------------
# 2. Index selection, HTTP retries, cache
# ---------------------------------------------------------------------------------------------

def test_directory_index_and_numeric_revision_selection(fixture_dir):
    files = ieso.parse_directory_index((fixture_dir / "ieso_index_excerpt.html").read_bytes())
    assert "PUB_GenOutputbyFuelHourly.xml" not in files  # rolling alias is never a dataset
    assert ieso.select_annual_file(files, 2024) == ("PUB_GenOutputbyFuelHourly_2024.xml", None)
    versioned = {"PUB_GenOutputbyFuelHourly_2026_v9.xml": (2026, 9), "PUB_GenOutputbyFuelHourly_2026_v10.xml": (2026, 10),
                 "PUB_GenOutputbyFuelHourly_2026_v263.xml": (2026, 263)}
    assert ieso.select_annual_file(versioned, 2026) == ("PUB_GenOutputbyFuelHourly_2026_v263.xml", 263)
    with pytest.raises(FileNotFoundError):
        ieso.select_annual_file(files, 1999)


class Scripted:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), 0

    def __call__(self, url, headers, ct, rt):
        self.calls += 1
        r = self.responses.pop(0)
        if isinstance(r, BaseException):
            raise r
        return r


def _client(transport, sleeps=None, attempts=5):
    return HttpClient(RetryPolicy(max_attempts=attempts, backoff_base=1.0, jitter=0.0), transport=transport,
                      sleep=(sleeps.append if sleeps is not None else (lambda s: None)))


def test_transient_failures_are_retried_with_exponential_backoff():
    sleeps = []
    t = Scripted([ConnectionResetError("reset"), TimeoutError("slow"), (200, {"content-length": "2"}, b"ok")])
    resp = _client(t, sleeps).request("https://x/y")
    assert resp.body == b"ok" and resp.n_requests == 3 and t.calls == 3
    assert sleeps == [1.0, 2.0]


def test_retries_are_bounded_to_five_attempts():
    t = Scripted([ConnectionResetError("reset")] * 10)
    with pytest.raises(TransientHttpError, match="5 attempts"):
        _client(t).request("https://x/y")
    assert t.calls == 5


def test_permanent_errors_fail_immediately_and_retry_after_is_honored():
    t = Scripted([(404, {}, b"nope")])
    with pytest.raises(HttpError, match="404"):
        _client(t).request("https://x/y")
    assert t.calls == 1
    sleeps = []
    t = Scripted([(429, {"retry-after": "7"}, b""), (503, {}, b""), (200, {}, b"ok")])
    assert _client(t, sleeps).request("https://x/y").body == b"ok"
    assert sleeps[0] == 7.0 and sleeps[1] == 2.0
    assert parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT", now=0) > 0 and parse_retry_after("soon") is None
    t = Scripted([(200, {"content-length": "10"}, b"short"), (200, {"content-length": "2"}, b"ok")])
    assert _client(t).request("https://x/y").n_requests == 2  # truncated body counts as transient


def test_chunked_download_assembles_validated_ranges_and_detects_changes():
    data = bytes(range(256)) * 40
    server = FakeServer({2024: data})
    url = ieso.BASE_URL + "PUB_GenOutputbyFuelHourly_2024.xml"
    resp = fake_client(server).download(url, chunk_size=1000)
    assert resp.body == data and resp.n_requests == math.ceil(len(data) / 1000)

    flaky = FakeServer({2024: data})
    calls = {"n": 0}

    def reset_every_other(url, headers, ct, rt):
        calls["n"] += 1
        if calls["n"] % 2 == 0:
            raise ConnectionResetError("reset mid-transfer")
        return flaky(url, headers, ct, rt)

    assert fake_client(reset_every_other).download(url, chunk_size=1000).body == data

    changing = FakeServer({2024: data})

    def modified_later(url, headers, ct, rt):
        status, h, body = changing(url, headers, ct, rt)
        if not headers["Range"].startswith("bytes=0-"):
            h = {**h, "last-modified": "Thu, 02 Jan 2025 00:00:00 GMT"}
        return status, h, body

    with pytest.raises(HttpError, match="changed"):
        fake_client(modified_later).download(url, chunk_size=1000)


def test_interrupted_cache_write_never_becomes_a_cache_hit(tmp_path, monkeypatch):
    store = cache.SnapshotStore(tmp_path)
    first = store.save("ieso/2024", "f.xml", b"old", {"url": "u"})
    real_replace = os.replace

    def crash(src, dst):
        if str(dst).endswith("f.xml"):
            raise KeyboardInterrupt("interrupted")
        return real_replace(src, dst)

    monkeypatch.setattr(cache.os, "replace", crash)
    with pytest.raises(KeyboardInterrupt):
        store.save("ieso/2024", "f.xml", b"new", {"url": "u"})
    monkeypatch.setattr(cache.os, "replace", real_replace)
    assert not list(tmp_path.rglob("*.partial"))
    latest = store.latest("ieso/2024")
    assert latest["snapshot"] == first["snapshot"] and store.load(latest) == b"old"

    store.path_of(latest).write_bytes(b"tampered")
    with pytest.raises(cache.CacheCorrupt):
        store.load(latest)


def test_offline_cache_miss_and_refresh_snapshots(tmp_path):
    from solar_test_utils import LOCATIONS

    loc = tmp_path / "loc.csv"
    LOCATIONS.to_csv(loc, index=False)
    with pytest.raises(cache.CacheMiss, match="offline"):
        collect("pilot", tmp_path / "d", loc, "2024-06-01", "2024-06-02", offline=True, log=lambda *a: None)

    server = FakeServer({2024: synthetic_ieso_xml(2024, pd.date_range("2024-06-01", "2024-06-02").date)})
    rec1, path = collect("pilot", tmp_path / "d", loc, "2024-06-01", "2024-06-02", client=fake_client(server),
                         log=lambda *a: None)
    n_calls = len(server.calls)
    rec2, _ = collect("pilot", tmp_path / "d", loc, "2024-06-01", "2024-06-02", offline=True, log=lambda *a: None)
    assert len(server.calls) == n_calls  # offline rebuild never touches the network
    assert rec1["ieso"]["files"][0]["sha256"] == rec2["ieso"]["files"][0]["sha256"]
    assert rec1["scale_label"] == "pilot-overridden" and path.name == "collection_pilot-overridden.json"
    rec3, _ = collect("pilot", tmp_path / "d", loc, "2024-06-01", "2024-06-02", refresh=True,
                      client=fake_client(server), log=lambda *a: None)
    old, new = rec1["ieso"]["files"][0], rec3["ieso"]["files"][0]
    assert new["snapshot"] != old["snapshot"] and new["sha256"] == old["sha256"]
    store = cache.SnapshotStore(tmp_path / "d" / "raw")
    assert store.load(old) == store.load(new)  # the old snapshot is untouched
    assert resolve_dates("paper") [2] == "paper"
    with pytest.raises(ValueError):
        collect("pilot", tmp_path / "d", loc, offline=True, refresh=True, log=lambda *a: None)


# ---------------------------------------------------------------------------------------------
# 3. Interval-time convention
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("day, hour, expected", [
    ("2024-01-01", 1, "2024-01-01T06:00:00+00:00"),
    ("2024-01-01", 24, "2024-01-02T05:00:00+00:00"),
    ("2024-07-15", 1, "2024-07-15T06:00:00+00:00"),        # same fixed offset in summer
    ("2024-03-10", 3, "2024-03-10T08:00:00+00:00"),        # spring-forward date: no DST shift
    ("2024-11-03", 2, "2024-11-03T07:00:00+00:00"),        # fall-back date: no repeated hour
    ("2024-02-29", 24, "2024-03-01T05:00:00+00:00"),       # leap day
    ("2024-12-31", 24, "2025-01-01T05:00:00+00:00"),       # year boundary
])
def test_hour_ending_maps_to_fixed_est_interval_end(day, hour, expected):
    assert ieso.interval_end_utc(day, hour).isoformat() == expected


def test_dst_dates_have_24_distinct_consecutive_intervals():
    for day in ("2024-03-10", "2024-11-03"):
        ends = [ieso.interval_end_utc(day, h) for h in range(1, 25)]
        assert len(set(ends)) == 24
        assert all((b - a) == dt.timedelta(hours=1) for a, b in zip(ends, ends[1:]))
    with pytest.raises(ValueError):
        ieso.interval_end_utc("2024-01-01", 0)


def test_normalized_generation_keeps_source_date_and_hour(fixture_dir):
    _, frame = _fixture_frame(fixture_dir)
    norm = ieso.normalize_generation(frame)
    r = norm[(norm["source_date"] == "2024-12-31") & (norm["source_hour"] == 24)].iloc[0]
    assert r["timestamp_utc"] == pd.Timestamp("2025-01-01T05:00Z")
    assert r["interval_start_utc"] == pd.Timestamp("2025-01-01T04:00Z")


# ---------------------------------------------------------------------------------------------
# 4. Open-Meteo
# ---------------------------------------------------------------------------------------------

PARAMS = open_meteo.request_params(43.6532, -79.3832, "2024-03-10", "2024-03-10")


def test_real_fixture_parses_with_units_grid_and_timestamps(fixture_dir):
    payload = json.loads((fixture_dir / "open_meteo_toronto_2024-03-10.json").read_text(encoding="utf-8"))
    meta, frame = open_meteo.parse_archive_json(payload, PARAMS)
    assert list(frame.columns) == ["timestamp_utc", *open_meteo.WEATHER_VARIABLES] and len(frame) == 24
    assert frame["timestamp_utc"].iloc[0] == pd.Timestamp("2024-03-10T00:00Z")
    assert (meta["resolved_latitude"], meta["resolved_longitude"], meta["elevation_m"]) == (43.75, -79.5, 99.0)
    assert meta["requested_model"] == "era5" and meta["units"]["wind_speed_10m"] == "km/h"
    assert frame.loc[16, "shortwave_radiation"] == 400.0  # value from the archived response
    assert PARAMS["timezone"] == "UTC" and PARAMS["models"] == "era5" and "_instant" not in PARAMS["hourly"]


def _payload(**changes):
    p = synthetic_weather_payload(PARAMS)
    for k, v in changes.items():
        p[k] = v
    return p


@pytest.mark.parametrize("mutate, message", [
    (lambda p: p["hourly_units"].__setitem__("wind_speed_10m", "m/s"), "unit of wind_speed_10m"),
    (lambda p: p["hourly"]["precipitation"].pop(), "array lengths"),
    (lambda p: p["hourly"].pop("cloud_cover"), "cloud_cover"),
    (lambda p: p.__setitem__("utc_offset_seconds", 3600), "utc_offset"),
    (lambda p: p["hourly"]["time"].__setitem__(3, "2024-03-10T09:00"), "contiguous"),
    (lambda p: p.__setitem__("latitude", 50.0), "far from request"),
    (lambda p: p.update(error=True, reason="bad"), "API error"),
])
def test_weather_validation_rejects_unit_length_and_time_problems(mutate, message):
    p = synthetic_weather_payload(PARAMS)
    mutate(p)
    with pytest.raises(open_meteo.OpenMeteoSchemaError, match=message):
        open_meteo.parse_archive_json(p, PARAMS)


def test_monthly_chunks_pad_and_boundary_duplicates_collapse():
    start, end = open_meteo.required_utc_range(dt.date(2024, 1, 30), dt.date(2024, 3, 1))
    chunks = open_meteo.monthly_chunks(start, end)
    assert [(m, str(a), str(b)) for m, a, b in chunks] == [
        ("2024-01", "2024-01-30", "2024-02-01"), ("2024-02", "2024-02-01", "2024-03-01"),
        ("2024-03", "2024-03-01", "2024-03-02")]
    frames = []
    for _, lo, hi in chunks:
        params = open_meteo.request_params(43.0, -80.0, lo, hi)
        frames.append(open_meteo.parse_archive_json(synthetic_weather_payload(params), params)[1])
    combined, collapsed = open_meteo.combine_chunks(frames)
    assert collapsed == 48 and combined["timestamp_utc"].is_unique
    assert len(combined) == ((end - start).days + 1) * 24
    frames[1].loc[0, "temperature_2m"] += 1.0  # 2024-02-01T00:00 now disagrees with chunk 1
    with pytest.raises(open_meteo.OpenMeteoSchemaError, match="conflicting"):
        open_meteo.combine_chunks(frames)


def test_equal_weight_aggregation_and_missing_location_behavior():
    grid = pd.date_range("2024-06-01T00:00Z", periods=3, freq="h")
    base = {name: [1.0, 2.0, 3.0] for name in open_meteo.WEATHER_VARIABLES}
    a = pd.DataFrame({"timestamp_utc": grid, **base})
    b = pd.DataFrame({"timestamp_utc": grid, **{k: [3.0, 6.0, np.nan] for k in base}})
    agg, reasons = open_meteo.aggregate_locations({"a": a, "b": b}, {"a": 0.5, "b": 0.5}, grid)
    assert agg["cloud_cover"].tolist()[:2] == [2.0, 4.0]
    assert np.isnan(agg["cloud_cover"].iloc[2])  # not renormalized onto location a
    assert reasons[0] == [] and "b:cloud_cover_missing" in reasons[2]
    bad = a.copy()
    bad.loc[1, "relative_humidity_2m"] = 101.0
    bad.loc[1, "shortwave_radiation"] = -1.0
    _, reasons = open_meteo.aggregate_locations({"a": bad, "b": a}, {"a": 0.5, "b": 0.5}, grid)
    assert set(reasons[1]) == {"a:relative_humidity_2m_out_of_range", "a:shortwave_radiation_out_of_range"}
    with pytest.raises(ValueError, match="configured"):
        open_meteo.aggregate_locations({"a": a}, {"a": 0.5, "b": 0.5}, grid)
    with pytest.raises(ValueError, match="sum to 1"):
        open_meteo.aggregate_locations({"a": a, "b": b}, {"a": 0.5, "b": 0.4}, grid)


def test_shipped_locations_file_has_five_equal_weights():
    from solar_test_utils import LOCATIONS_FILE

    table = open_meteo.read_locations(LOCATIONS_FILE)
    assert table["location_id"].tolist() == ["windsor", "london", "toronto", "kingston", "ottawa"]
    assert (table["weight"] == 0.2).all()
