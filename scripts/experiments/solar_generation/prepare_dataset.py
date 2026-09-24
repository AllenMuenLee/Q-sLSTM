# scripts/experiments/solar_generation/prepare_dataset.py
#
# Build the frozen processed dataset (hourly table, windows, scaler, quality report, manifest) from
# the pinned raw snapshots of a collection, and write data/solar_generation/prepared_<scale_label>.json.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from q_slstm.data_sources.collect import collect, collection_path  # noqa: E402
from q_slstm.datasets.solar_generation import DEFAULT_MIN_COVERAGE, prepare_dataset  # noqa: E402

DEFAULT_LOCATIONS = ROOT / "configs" / "solar_generation" / "locations.csv"


def build_parser():
    p = argparse.ArgumentParser(description="Prepare the solar-generation dataset from collected raw data.")
    p.add_argument("--scale", choices=["paper", "pilot"], default="paper")
    p.add_argument("--data-dir", default="data/solar_generation")
    p.add_argument("--collection", default=None,
                   help="collection record (default: <data-dir>/collection_<scale>.json)")
    p.add_argument("--sequence-length", type=int, default=None, help="input window L (paper 32, pilot 8)")
    p.add_argument("--min-coverage", type=float, default=DEFAULT_MIN_COVERAGE,
                   help="minimum valid-row fraction per split (default 0.95; overrides are recorded)")
    p.add_argument("--offline", action="store_true",
                   help="never download; without it a missing collection record is collected first")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.collection is None and not collection_path(args.data_dir, args.scale).exists():
        if args.offline:
            raise SystemExit(f"no collection record for {args.scale} and --offline given; run collect_data.py")
        collect(args.scale, args.data_dir, DEFAULT_LOCATIONS)
    prepare_dataset(args.data_dir, args.scale, args.collection, args.sequence_length, args.min_coverage)
    return 0


if __name__ == "__main__":
    sys.exit(main())
