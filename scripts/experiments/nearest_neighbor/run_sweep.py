# scripts/experiments/nearest_neighbor/run_sweep.py
#
# Run ONE model (qlstm or qslstm) over many seeds, with a chosen number of parallel workers.
# Run the script once per model; combine afterwards with analyze_results.py.
#
# Pairing across the two invocations comes from the seeds: the same seed always gives identical
# dataset tensors, splits, loader order, and initial shared VQC/output parameters (see
# derive_seeds). Random seeds are drawn reproducibly from --master-seed, so running the other model
# with the same --n-seeds and --master-seed (or the same --seeds list) reuses exactly the same seeds.
# Each run is a separate train_nearest_neighbor.py process so a failure cannot corrupt another run.

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
sys.path.insert(0, str(SCRIPT_DIR))

from q_slstm.experiments.nearest_neighbor import PRESETS, add_run_arguments, resolve_config, run_directory  # noqa: E402
from q_slstm.models.factory import QUANTUM_MODELS  # noqa: E402
from q_slstm.models.q_slstm_cell import QSLSTM_RECURRENCE  # noqa: E402
from q_slstm.models.q_slstm_log_cell import QSLSTM_LOG_RECURRENCE  # noqa: E402
from q_slstm.utils.seeds import draw_seeds  # noqa: E402

SWEEP_ONLY = {"model", "seeds", "n_seeds", "master_seed", "workers", "resume", "dry_run"}


def build_parser():
    parser = argparse.ArgumentParser(description="Run one model (qlstm or qslstm) over many seeds.")
    parser.add_argument("--model", choices=QUANTUM_MODELS, required=True, help="the single model to run")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--n-seeds", type=int, default=None,
                       help="number of random seeds, drawn reproducibly from --master-seed "
                            "(default: the scale preset's count)")
    group.add_argument("--seeds", type=int, nargs="+", default=None, help="explicit ordered run seeds")
    parser.add_argument("--master-seed", type=int, default=0,
                        help="seed of the seed generator; use the same value for both models (default: 0)")
    parser.add_argument("--workers", "--jobs", dest="workers", type=int, default=1,
                        help="number of runs executing in parallel (default: 1, sequential)")
    parser.add_argument("--resume", action="store_true", help="skip runs whose complete.json already exists")
    parser.add_argument("--dry-run", action="store_true", help="print the planned runs and exit")
    add_run_arguments(parser)
    return parser


def child_arguments(args, model, seed):
    argv = ["--model", model, "--seed", str(seed)]
    for key, value in vars(args).items():
        if key in SWEEP_ONLY or value is None or value is False:
            continue
        flag = "--" + key.replace("_", "-")
        argv += [flag] if value is True else [flag, str(value)]
    return argv


def run_one(job):
    argv, run_dir, resume = job
    if (run_dir / "config.json").exists():
        previous = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        model_name = argv[argv.index("--model") + 1]
        recurrence = QSLSTM_LOG_RECURRENCE if model_name == "qslstm_log" else QSLSTM_RECURRENCE
        if previous.get("qslstm_recurrence") != recurrence:
            return run_dir, "FAILED (different recurrence; choose a new --save-dir)", 0.0
    if resume and (run_dir / "complete.json").exists():
        return run_dir, "skipped", 0.0
    run_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with open(run_dir / "console_log.txt", "w", encoding="utf-8") as log:
        code = subprocess.run([sys.executable, str(SCRIPT_DIR / "train_nearest_neighbor.py"), *argv],
                              stdout=log, stderr=subprocess.STDOUT).returncode
    return run_dir, "ok" if code == 0 else f"FAILED (exit {code})", time.perf_counter() - start


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.seeds is not None:
        seeds = args.seeds
        if len(set(seeds)) != len(seeds):
            raise SystemExit(f"--seeds must not repeat values, got {seeds}")
    else:
        n = args.n_seeds if args.n_seeds is not None else PRESETS[args.scale]["n_seeds"]
        if n < 1:
            raise SystemExit("--n-seeds must be at least 1")
        seeds = draw_seeds(n, args.master_seed)

    jobs, scale_label = [], None
    for seed in seeds:
        config = resolve_config({**vars(args), "model": args.model, "seed": seed})
        scale_label = config["scale_label"]
        jobs.append((child_arguments(args, args.model, seed), run_directory(config), args.resume))

    print(f"scale={scale_label} model={args.model}: {len(jobs)} runs, {args.workers} worker(s)", flush=True)
    print(f"seeds: {seeds}", flush=True)
    for _, run_dir, _ in jobs:
        print("  ", run_dir, flush=True)
    if args.dry_run:
        return 0

    root = Path(args.save_dir) / scale_label
    root.mkdir(parents=True, exist_ok=True)
    (root / f"seeds_{args.model}.json").write_text(json.dumps(
        {"model": args.model, "seeds": seeds, "n_seeds": len(seeds),
         "master_seed": None if args.seeds is not None else args.master_seed}, indent=2), encoding="utf-8")

    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for run_dir, status, seconds in pool.map(run_one, jobs):
            print(f"{status:>14}  {seconds:8.1f}s  {run_dir}", flush=True)
            failed += status.startswith("FAILED")
    if failed:
        print(f"{failed} run(s) failed; see failure.json / console_log.txt in the run directories", file=sys.stderr)
    else:
        print(f"done. After running the other model with the same seeds: "
              f"python scripts/experiments/nearest_neighbor/analyze_results.py --runs-dir {root}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
