# scripts/experiments/solar_generation/run_sweep.py
#
# Run ONE model (qlstm or qslstm) over many seeds with a chosen number of parallel workers; run the
# script once per model with the same seeds, then combine with analyze_results.py. The dataset and
# chronological split are fixed, so seeds only change initialization and training order. Each run is
# an isolated train_solar_generation.py subprocess with its own console_log.txt.

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(ROOT / "src"))

import pandas  # noqa: E402,F401  (import before torch: avoids a Windows pyarrow/torch DLL crash)

from q_slstm.models.factory import QUANTUM_MODELS  # noqa: E402
from q_slstm.utils.seeds import draw_seeds  # noqa: E402
from q_slstm.data_sources.cache import atomic_write_json  # noqa: E402
from q_slstm.experiments.solar_generation import (  # noqa: E402
    PRESETS, add_run_arguments, resolve_config, run_directory, run_status, study_directory)

SWEEP_ONLY = {"model", "seeds", "n_seeds", "master_seed", "workers", "resume", "overwrite", "dry_run"}


def build_parser():
    parser = argparse.ArgumentParser(description="Run one model (qlstm or qslstm) over many seeds.")
    parser.add_argument("--model", choices=QUANTUM_MODELS, required=True, help="the single model to run")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--n-seeds", type=int, default=None,
                       help="number of seeds drawn reproducibly from --master-seed (default: preset count)")
    group.add_argument("--seeds", type=int, nargs="+", default=None, help="explicit ordered run seeds")
    parser.add_argument("--master-seed", type=int, default=0, help="seed of the seed generator (default 0)")
    parser.add_argument("--workers", "--jobs", dest="workers", type=int, default=1,
                        help="runs executing in parallel (default 1)")
    parser.add_argument("--resume", action="store_true",
                        help="skip verified complete runs; continue interrupted ones from last.pt")
    parser.add_argument("--overwrite", action="store_true", help="rerun complete runs")
    parser.add_argument("--dry-run", action="store_true", help="print the planned runs and exit")
    add_run_arguments(parser)
    return parser


def child_arguments(args, seed):
    argv = ["--model", args.model, "--seed", str(seed)]
    for key, value in vars(args).items():
        if key in SWEEP_ONLY or value is None or value is False:
            continue
        flag = "--" + key.replace("_", "-")
        argv += [flag] if value is True else [flag, str(value)]
    if args.resume:
        argv.append("--resume")
    if args.overwrite:
        argv.append("--overwrite")
    return argv


def run_one(job):
    argv, run_dir, status = job
    if status == "skip":
        return run_dir, "skipped (complete)", 0.0
    run_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with open(run_dir / "console_log.txt", "a" if status == "incomplete" else "w", encoding="utf-8") as log:
        code = subprocess.run([sys.executable, str(SCRIPT_DIR / "train_solar_generation.py"), *argv],
                              stdout=log, stderr=subprocess.STDOUT).returncode
    return run_dir, "ok" if code == 0 else f"FAILED (exit {code})", time.perf_counter() - start


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.resume and args.overwrite:
        raise SystemExit("--resume and --overwrite are mutually exclusive")
    if args.seeds is not None:
        seeds = args.seeds
        if len(set(seeds)) != len(seeds):
            raise SystemExit(f"--seeds must not repeat values, got {seeds}")
    else:
        n = args.n_seeds if args.n_seeds is not None else PRESETS[args.scale]["n_seeds"]
        if n < 1:
            raise SystemExit("--n-seeds must be at least 1")
        seeds = draw_seeds(n, args.master_seed)

    jobs, configs, blocked = [], [], []
    for seed in seeds:
        config = resolve_config({**vars(args), "seed": seed})
        run_dir, status = run_directory(config), run_status(config)
        if status == "incompatible" and not args.overwrite:
            blocked.append(f"{run_dir}: incompatible existing run")
        if status == "complete" and not (args.resume or args.overwrite):
            blocked.append(f"{run_dir}: already complete (use --resume to skip or --overwrite to rerun)")
        jobs.append((child_arguments(args, seed), run_dir,
                     "skip" if status == "complete" and args.resume else status))
        configs.append(config)
    study_dir = study_directory(configs[0])

    print(f"study={configs[0]['study_id']} scale={configs[0]['study']['scale_label']} model={args.model}: "
          f"{len(jobs)} runs, {args.workers} worker(s)", flush=True)
    print(f"seeds: {json.dumps(seeds)}", flush=True)
    for _, run_dir, status in jobs:
        print(f"   {run_dir}  [{status}]", flush=True)
    if blocked:
        raise SystemExit("refusing to overwrite:\n  " + "\n  ".join(blocked))
    if args.dry_run:
        return 0

    study_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = study_dir / "study_manifest.json"
    study_manifest = {"study_id": configs[0]["study_id"], "study": configs[0]["study"],
                      "dataset_manifest": configs[0]["dataset_manifest"]}
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing["study"] != study_manifest["study"]:
            raise SystemExit(f"{manifest_path} describes a different study")
    else:
        atomic_write_json(manifest_path, study_manifest)
    atomic_write_json(study_dir / f"seeds_{args.model}.json", {
        "model": args.model, "seeds": seeds, "n_seeds_requested": len(seeds),
        "master_seed": None if args.seeds is not None else args.master_seed})

    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for run_dir, status, seconds in pool.map(run_one, jobs):
            print(f"{status:>20}  {seconds:8.1f}s  {run_dir}", flush=True)
            failed += status.startswith("FAILED")
    print(f"study directory: {study_dir}")
    if failed:
        print(f"{failed} run(s) failed; see failure.json / console_log.txt in the run directories", file=sys.stderr)
    print("after both models have run with the same seeds:\n"
          f"python scripts/experiments/solar_generation/analyze_results.py --runs-dir {study_dir}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
