# IESO "Generator Output by Fuel Type Hourly" adapter (SOLAR only).
#
# Source: https://reports-public.ieso.ca/public/GenOutputbyFuelHourly/
# Report description: https://reports.ieso.ca/docrefs/helpfile/GenOutputbyFuelHourly_h1.pdf
#   unit MWh; generators >= 20 MW registered with the IESO; telemetry issues affect aggregates.
# Schema: https://reports-public.ieso.ca/docrefs/schema/GenOutputbyFuelHourly_r1.xsd
#
# Source-to-column mapping (namespace http://www.ieso.ca/schema):
#   DocBody/DailyData/Day                                     -> source_date   (YYYY-MM-DD, EST)
#   DocBody/DailyData/HourlyData/Hour                         -> source_hour   (hour ending, 1..24)
#   .../FuelTotal[Fuel='SOLAR']/EnergyValue/Output            -> solar_generation (MWh, unchanged)
#   .../FuelTotal[Fuel='SOLAR']/EnergyValue/OutputQuality     -> output_quality (0 = complete;
#       negative = number of generator data points not available; the IESO stylesheet renders
#       values with OutputQuality < 0 in red)
# EnergyValue and Output are optional in the schema: an absent SOLAR record, EnergyValue, or Output
# is a missing observation (NaN), never zero production.
#
# Time convention: IESO market reports use Eastern Standard Time all year. Every day in the annual
# files has exactly 24 hours, including DST-transition dates, consistent with fixed UTC-05:00.
#   end_est = midnight(D, UTC-05:00) + H hours; timestamp_utc = end_est in UTC.

from __future__ import annotations

import datetime as dt
import math
import re
import xml.etree.ElementTree as ET

import pandas as pd

BASE_URL = "https://reports-public.ieso.ca/public/GenOutputbyFuelHourly/"
HELP_URL = "https://reports.ieso.ca/docrefs/helpfile/GenOutputbyFuelHourly_h1.pdf"
SCHEMA_URL = "https://reports-public.ieso.ca/docrefs/schema/GenOutputbyFuelHourly_r1.xsd"
NS = "http://www.ieso.ca/schema"
DOC_ID = "GenOutputbyFuelHourly"
FUEL = "SOLAR"
UNIT = "MWh"
EST = dt.timezone(dt.timedelta(hours=-5), "EST")
ATTRIBUTION = "Independent Electricity System Operator (IESO), Generator Output by Fuel Type Hourly Report"

_ANNUAL = re.compile(r"PUB_GenOutputbyFuelHourly_(\d{4})(?:_v(\d+))?\.xml$")

STATUS_OK, STATUS_FLAGGED, STATUS_MISSING = "ok", "flagged", "missing"


class IesoSchemaError(ValueError):
    """The payload does not have the expected report structure."""


def _q(tag):
    return f"{{{NS}}}{tag}"


# ---------------------------------------------------------------------------------------------
# Directory index and snapshot selection
# ---------------------------------------------------------------------------------------------

def parse_directory_index(html):
    """Annual report files listed in the directory index: {filename: (year, revision or None)}.

    The rolling alias PUB_GenOutputbyFuelHourly.xml is deliberately not matched.
    """
    if isinstance(html, bytes):
        html = html.decode("utf-8", "replace")
    files = {}
    for name in re.findall(r'href="([^"]+\.xml)"', html):
        name = name.rsplit("/", 1)[-1]
        match = _ANNUAL.match(name)
        if match:
            files[name] = (int(match.group(1)), int(match.group(2)) if match.group(2) else None)
    return files


def select_annual_file(files, year):
    """One authoritative file for `year`: the unversioned annual file, else the highest revision."""
    candidates = {name: rev for name, (y, rev) in files.items() if y == year}
    if not candidates:
        raise FileNotFoundError(f"no IESO annual file for {year} in the directory index")
    unversioned = [name for name, rev in candidates.items() if rev is None]
    if unversioned:
        return unversioned[0], None
    name = max(candidates, key=lambda n: candidates[n])
    return name, candidates[name]


# ---------------------------------------------------------------------------------------------
# Time mapping
# ---------------------------------------------------------------------------------------------

def interval_end_utc(source_date, hour_ending):
    """UTC interval end of IESO date `source_date` (EST) and hour ending `hour_ending` (1..24)."""
    if not 1 <= int(hour_ending) <= 24:
        raise ValueError(f"hour ending must be in 1..24, got {hour_ending}")
    if isinstance(source_date, str):
        source_date = dt.date.fromisoformat(source_date)
    midnight = dt.datetime.combine(source_date, dt.time(0), tzinfo=EST)
    return (midnight + dt.timedelta(hours=int(hour_ending))).astimezone(dt.timezone.utc)


# ---------------------------------------------------------------------------------------------
# XML parsing
# ---------------------------------------------------------------------------------------------

_ALLOWED_CHILDREN = {
    "DocBody": {"DeliveryYear", "DailyData"},
    "DailyData": {"Day", "HourlyData"},
    "HourlyData": {"Hour", "FuelTotal"},
    "FuelTotal": {"Fuel", "EnergyValue"},
    "EnergyValue": {"OutputQuality", "Output"},
}


def _check_children(elem, kind, where):
    for child in elem:
        if not child.tag.startswith(f"{{{NS}}}"):
            raise IesoSchemaError(f"{where}: element {child.tag!r} outside namespace {NS}")
        name = child.tag.split("}", 1)[1]
        if name not in _ALLOWED_CHILDREN[kind]:
            raise IesoSchemaError(f"{where}: unexpected element <{name}> in <{kind}> (schema change?)")


def _text(elem, tag, where, required=True):
    found = elem.findall(_q(tag))
    if len(found) > 1:
        raise IesoSchemaError(f"{where}: repeated <{tag}>")
    if not found or found[0].text is None or not found[0].text.strip():
        if required:
            raise IesoSchemaError(f"{where}: missing <{tag}>")
        return None
    return found[0].text.strip()


def _number(text, where, what, integer=False):
    try:
        value = int(text) if integer else float(text)
    except ValueError:
        raise IesoSchemaError(f"{where}: non-numeric {what} {text!r}") from None
    if not integer and not math.isfinite(value):
        raise IesoSchemaError(f"{where}: non-finite {what} {text!r}")
    return value


def parse_generation_xml(data, source_file="<memory>"):
    """Parse one annual report into (header dict, per-hour SOLAR DataFrame).

    Columns: source_date, source_hour, solar_generation (MWh, NaN when missing), output_quality
    (Int64, <NA> when absent), fuel_record_present, energy_value_present, output_present, status.
    Duplicate records collapse when identical and fail when conflicting.
    """
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise IesoSchemaError(f"{source_file}: malformed XML: {exc}") from exc
    if root.tag != _q("Document"):
        raise IesoSchemaError(f"{source_file}: root element is {root.tag!r}, expected {{{NS}}}Document")
    if root.get("docID") != DOC_ID:
        raise IesoSchemaError(f"{source_file}: docID {root.get('docID')!r}, expected {DOC_ID!r}")

    header_elem = root.find(_q("DocHeader"))
    header = {"doc_id": DOC_ID}
    if header_elem is not None:
        for tag, key in (("DocTitle", "doc_title"), ("DocRevision", "doc_revision"), ("CreatedAt", "created_at")):
            header[key] = _text(header_elem, tag, source_file, required=False)
    body = root.find(_q("DocBody"))
    if body is None:
        raise IesoSchemaError(f"{source_file}: missing <DocBody>")
    _check_children(body, "DocBody", source_file)
    year = _number(_text(body, "DeliveryYear", source_file), source_file, "DeliveryYear", integer=True)
    header["delivery_year"] = year

    records = {}
    duplicates_collapsed = 0
    for d_i, day_elem in enumerate(body.findall(_q("DailyData"))):
        where = f"{source_file} DailyData[{d_i}]"
        _check_children(day_elem, "DailyData", where)
        day_text = _text(day_elem, "Day", where)
        try:
            day = dt.date.fromisoformat(day_text)
        except ValueError:
            raise IesoSchemaError(f"{where}: invalid Day {day_text!r}") from None
        if day.year != year:
            raise IesoSchemaError(f"{where}: Day {day} outside DeliveryYear {year}")
        for h_i, hour_elem in enumerate(day_elem.findall(_q("HourlyData"))):
            hwhere = f"{where} ({day}) HourlyData[{h_i}]"
            _check_children(hour_elem, "HourlyData", hwhere)
            hour = _number(_text(hour_elem, "Hour", hwhere), hwhere, "Hour", integer=True)
            if not 1 <= hour <= 24:
                raise IesoSchemaError(f"{hwhere}: Hour {hour} outside 1..24")
            solar = []
            for fuel_elem in hour_elem.findall(_q("FuelTotal")):
                _check_children(fuel_elem, "FuelTotal", hwhere)
                if _text(fuel_elem, "Fuel", hwhere) == FUEL:
                    solar.append(fuel_elem)
            if len(solar) > 1:
                raise IesoSchemaError(f"{hwhere}: {len(solar)} SOLAR records in one hour")
            record = _solar_record(solar[0] if solar else None, f"{hwhere} hour {hour}")
            key = (day, hour)
            if key in records:
                if not _same_record(records[key], record):
                    raise IesoSchemaError(f"{source_file}: conflicting duplicate SOLAR records for {day} hour {hour}")
                duplicates_collapsed += 1
                continue
            records[key] = record

    header["duplicates_collapsed"] = duplicates_collapsed
    rows = [{"source_date": day.isoformat(), "source_hour": hour, **rec} for (day, hour), rec in sorted(records.items())]
    frame = pd.DataFrame(rows, columns=["source_date", "source_hour", "solar_generation", "output_quality",
                                        "fuel_record_present", "energy_value_present", "output_present", "status"])
    frame["output_quality"] = frame["output_quality"].astype("Int64")
    frame["solar_generation"] = frame["solar_generation"].astype(float)
    return header, frame


def _solar_record(fuel_elem, where):
    rec = {"solar_generation": math.nan, "output_quality": pd.NA, "fuel_record_present": fuel_elem is not None,
           "energy_value_present": False, "output_present": False, "status": STATUS_MISSING}
    if fuel_elem is None:
        return rec
    energy = fuel_elem.findall(_q("EnergyValue"))
    if len(energy) > 1:
        raise IesoSchemaError(f"{where}: repeated <EnergyValue>")
    if not energy:
        return rec
    _check_children(energy[0], "EnergyValue", where)
    rec["energy_value_present"] = True
    rec["output_quality"] = _number(_text(energy[0], "OutputQuality", where), where, "OutputQuality", integer=True)
    output = _text(energy[0], "Output", where, required=False)
    if output is not None:
        rec["solar_generation"] = _number(output, where, "Output")
        rec["output_present"] = True
        rec["status"] = STATUS_OK if rec["output_quality"] == 0 else STATUS_FLAGGED
    return rec


def _same_record(a, b):
    def norm(r):
        return tuple("nan" if isinstance(v, float) and math.isnan(v) else ("na" if v is pd.NA else v)
                     for v in (r["solar_generation"], r["output_quality"], r["fuel_record_present"],
                               r["energy_value_present"], r["output_present"]))
    return norm(a) == norm(b)


def normalize_generation(frame):
    """Add interval_end (timestamp_utc) and interval_start_utc columns to a parsed SOLAR frame."""
    frame = frame.copy()
    ends = [interval_end_utc(d, h) for d, h in zip(frame["source_date"], frame["source_hour"])]
    frame.insert(0, "timestamp_utc", pd.to_datetime(ends, utc=True))
    frame.insert(1, "interval_start_utc", frame["timestamp_utc"] - pd.Timedelta(hours=1))
    return frame
