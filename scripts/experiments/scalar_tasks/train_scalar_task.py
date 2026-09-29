# scripts/experiments/scalar_tasks/train_scalar_task.py
#
# Train and evaluate one model on one seed of one 1-D scalar sequence task.
# Use run_sweep.py to run a model over many seeds and tasks.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from q_slstm.datasets.scalar_tasks import TASKS  # noqa: E402
from q_slstm.experiments.scalar_tasks import ALL_TASKS  # noqa: E402
from q_slstm.experiments.scalar_tasks import MODELS, add_run_arguments, resolve_config, run_experiment  # noqa: E402


def build_parser():
    parser = argparse.ArgumentParser(description="1-D scalar sequence task experiment (one run).")
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--task", choices=ALL_TASKS, required=True)
    parser.add_argument("--seed", type=int, default=0, help="run seed; paired models share it")
    add_run_arguments(parser)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = resolve_config(args)
    print(f"scale={config['scale_label']} task={config['task']} model={config['model']} "
          f"seed={config['seeds']['run_seed']} L={config['sequence_length']} "
          f"train/val/test={config['optimizer_train_size']}/{config['val_size']}/{config['test_size']} "
          f"hidden={config['hidden_size']} depth={config['qnn_depth']}", flush=True)
    run_dir = run_experiment(config)
    print(f"run complete: {run_dir}")


if __name__ == "__main__":
    main()
