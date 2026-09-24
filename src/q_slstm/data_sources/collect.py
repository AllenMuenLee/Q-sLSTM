# Data acquisition: IESO SOLAR generation + Open-Meteo ERA5 weather at the frozen proxy locations.
#
# Writes raw snapshots (see cache.py) and a collection record `collection_<scale_label>.json` that
# pins exactly which snapshot bytes (by SHA-256) the dataset preparation must use. `offline=True`
# never touches the network; `refresh=True` downloads new snapshots without modifying old ones.

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from q_slstm.data_sources import ieso, open_meteo
from q_slstm.data_sources.cache import CacheMiss, SnapshotStore, atomic_write_json, sha256_file, utc_now
from q_slstm.data_sources.http import HttpClient

SCALE_DATES = {
    "paper": (dt.date(2024, 1, 1), dt.date(2025, 12, 31)),
    "pilot": (dt.date(2024, 6, 1), dt.date(2024, 6, 14)),
}
COLLECTION_SCHEMA = "solar_generation_collection/1"


def collection_path(data_dir, scale_label):
    return Path(data_dir) / f"collection_{scale_label}.json"


def resolve_dates(scale, start_date=None, end_date=None):
    """(start, end, scale_label, overrides) for a scale preset and optional explicit dates."""
    if scale not in SCALE_DATES:
        raise ValueError(f"unknown scale {scale!r}")
    preset_start, preset_end = SCALE_DATES[scale]
    start = dt.date.fromisoformat(str(start_date)) if start_date else preset_start
    end = dt.date.fromisoformat(str(end_date)) if end_date else preset_end
    if end < start:
        raise ValueError(f"end date {end} precedes start date {start}")
    overrides = {}
    if start != preset_start:
        overrides["start_date"] = {"preset": str(preset_start), "used": str(start)}
    if end != preset_end:
        overrides["end_date"] = {"preset": str(preset_end), "used": str(end)}
    return start, end, scale if not overrides else f"{scale}-overridden", overrides


class Collector:
    def __init__(self, data_dir, client=None, offline=False, refresh=False, log=print):
        if offline and refresh:
            raise ValueError("--offline and --refresh are mutually exclusive")
        self.store = SnapshotStore(Path(data_dir) / "raw")
        self.client = client or HttpClient()
        self.offline, self.refresh, self.log = offline, refresh, log

    def _cached(self, key, matches=None):
        record = self.store.latest(key)
        if record is not None and (matches is None or matches(record)):
            self.store.load(record)  # re-verify bytes before trusting the cache hit
            return record
        return None

    def _fetch(self, key, filename, fetch, metadata, matches=None):
        if not self.refresh:
            record = self._cached(key, matches)
            if record is not None:
                self.log(f"  cached   {key} ({record['snapshot']})")
                return record, False
        if self.offline:
            raise CacheMiss(f"offline mode: no cached snapshot for {key}; run collect_data.py online first")
        response = fetch()
        record = self.store.save(key, filename, response.body, {
            **metadata,
            "http_status": response.status,
            "etag": response.headers.get("etag"),
            "last_modified": response.headers.get("last-modified"),
            "content_type": response.headers.get("content-type"),
            "n_http_requests": response.n_requests,
            "n_http_retries": response.n_retries,
        })
        self.log(f"  fetched  {key} ({record['n_bytes']} bytes, {response.n_requests} request(s))")
        return record, True

    # -- IESO -----------------------------------------------------------------------------------

    def ieso_files(self, years):
        index, _ = self._fetch("ieso/index", "index.html", lambda: self.client.request(ieso.BASE_URL),
                               {"source": "ieso", "url": ieso.BASE_URL})
        listing = ieso.parse_directory_index(self.store.load(index))
        records = []
        for year in years:
            filename, revision = ieso.select_annual_file(listing, year)
            url = ieso.BASE_URL + filename
            record, _ = self._fetch(
                f"ieso/{year}", filename, lambda url=url: self.client.download(url),
                {"source": "ieso", "url": url, "year": year, "revision": revision,
                 "selection_rule": "unversioned annual file, else highest numeric revision"},
                matches=lambda r, filename=filename: r["filename"] == filename)
            header, frame = ieso.parse_generation_xml(self.store.load(record), filename)
            if header["delivery_year"] != year:
                raise ieso.IesoSchemaError(f"{filename}: DeliveryYear {header['delivery_year']} != {year}")
            records.append({**record, "doc_header": header, "n_hour_records": int(len(frame))})
        return {"index": index, "files": records}

    # -- Open-Meteo -----------------------------------------------------------------------------

    def weather_files(self, locations, start, end):
        utc_start, utc_end = open_meteo.required_utc_range(start, end)
        out = {}
        for row in locations.itertuples(index=False):
            records = []
            for month, lo, hi in open_meteo.monthly_chunks(utc_start, utc_end):
                params = open_meteo.request_params(row.latitude, row.longitude, lo, hi)
                key = f"open_meteo/{row.location_id}/{lo}_{hi}"
                record, _ = self._fetch(
                    key, f"{row.location_id}_{lo}_{hi}.json",
                    lambda params=params: self.client.request(open_meteo.ARCHIVE_URL, params),
                    {"source": "open_meteo", "url": open_meteo.ARCHIVE_URL, "params": params,
                     "location_id": row.location_id, "month": month},
                    matches=lambda r, params=params: r.get("params") == params)
                meta, _ = open_meteo.parse_archive_json(json.loads(self.store.load(record)), params, key)
                records.append({**record, "response_metadata": meta})
            out[row.location_id] = records
        return out


def collect(scale, data_dir, locations_file, start_date=None, end_date=None, offline=False, refresh=False,
            client=None, log=print):
    """Acquire (or verify cached) raw data for one scale and write its collection record."""
    start, end, scale_label, overrides = resolve_dates(scale, start_date, end_date)
    locations = open_meteo.read_locations(locations_file)
    collector = Collector(data_dir, client=client, offline=offline, refresh=refresh, log=log)
    log(f"collecting {scale_label}: source dates {start}..{end} (EST), {len(locations)} weather locations")
    ieso_part = collector.ieso_files(range(start.year, end.year + 1))
    weather_part = collector.weather_files(locations, start, end)
    record = {
        "schema": COLLECTION_SCHEMA,
        "scale": scale,
        "scale_label": scale_label,
        "date_overrides": overrides,
        "start_date": str(start),
        "end_date": str(end),
        "collected_at_utc": utc_now().isoformat().replace("+00:00", "Z"),
        "offline": offline,
        "refresh": refresh,
        "http_policy": collector.client.policy.to_dict(),
        "locations": locations.to_dict(orient="records"),
        "locations_file_sha256": sha256_file(locations_file),
        "ieso": ieso_part,
        "open_meteo": weather_part,
        "attribution": [ieso.ATTRIBUTION, open_meteo.ATTRIBUTION],
        "documentation": {"ieso_report": ieso.HELP_URL, "ieso_schema": ieso.SCHEMA_URL,
                          "ieso_directory": ieso.BASE_URL, "open_meteo": open_meteo.DOCS_URL},
    }
    path = collection_path(data_dir, scale_label)
    atomic_write_json(path, record)
    log(f"collection record: {path}")
    return record, path
