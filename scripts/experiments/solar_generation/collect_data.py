# scripts/experiments/solar_generation/collect_data.py
#
# Download (or verify cached) IESO SOLAR generation and Open-Meteo ERA5 weather for one scale and
# write data/solar_generation/collection_<scale_label>.json pinning the raw snapshot bytes.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from q_slstm.data_sources.collect import collect  # noqa: E402
from q_slstm.data_sources.http import HttpClient, RetryPolicy  # noqa: E402

DEFAULT_LOCATIONS = ROOT / "configs" / "solar_generation" / "locations.csv"


def build_parser():
    p = argparse.ArgumentParser(description="Collect raw data for the solar-generation experiment.")
    p.add_argument("--scale", choices=["paper", "pilot"], default="paper")
    p.add_argument("--start-date", default=None, help="first IESO source date (EST), default from --scale")
    p.add_argument("--end-date", default=None, help="last IESO source date (EST, inclusive), default from --scale")
    p.add_argument("--locations-file", default=str(DEFAULT_LOCATIONS))
    p.add_argument("--data-dir", default="data/solar_generation")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--offline", action="store_true", help="use cached snapshots only; never touch the network")
    mode.add_argument("--refresh", action="store_true", help="download new snapshots (old ones are kept)")
    p.add_argument("--max-attempts", type=int, default=5, help="attempts per HTTP request (default 5)")
    p.add_argument("--connect-timeout", type=float, default=15.0)
    p.add_argument("--read-timeout", type=float, default=60.0)
    p.add_argument("--min-request-interval", type=float, default=0.5,
                   help="seconds between consecutive requests (provider politeness, default 0.5)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    policy = RetryPolicy(max_attempts=args.max_attempts, connect_timeout=args.connect_timeout,
                         read_timeout=args.read_timeout, min_request_interval=args.min_request_interval)
    collect(args.scale, args.data_dir, args.locations_file, args.start_date, args.end_date,
            offline=args.offline, refresh=args.refresh, client=HttpClient(policy))
    return 0


if __name__ == "__main__":
    sys.exit(main())
