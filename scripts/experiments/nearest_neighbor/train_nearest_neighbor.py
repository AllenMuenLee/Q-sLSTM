# scripts/experiments/nearest_neighbor/train_nearest_neighbor.py
#
# Train and evaluate one model (conventional QLSTM or stabilized Q-sLSTM) on one seed of the
# nearest-neighbor memory-revision task. Use run_sweep.py to run both models over paired seeds.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from q_slstm.experiments.nearest_neighbor import add_run_arguments, resolve_config, run_experiment  # noqa: E402
from q_slstm.models.factory import QUANTUM_MODELS  # noqa: E402


def build_parser():
    parser = argparse.ArgumentParser(description="Nearest-neighbor memory-revision experiment (one run).")
    parser.add_argument("--model", choices=QUANTUM_MODELS, required=True)
    parser.add_argument("--seed", type=int, default=0, help="run seed; paired models share it")
    add_run_arguments(parser)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = resolve_config(args)
    print(f"scale={config['scale_label']} model={config['model']} seed={config['seeds']['run_seed']} "
          f"L={config['sequence_length']} ({config['n_candidates']} candidates) "
          f"train/val/test={config['optimizer_train_size']}/{config['val_size']}/{config['test_size']} "
          f"qubits={config['n_qubits']} depth={config['qnn_depth']}", flush=True)
    run_dir = run_experiment(config)
    print(f"run complete: {run_dir}")


if __name__ == "__main__":
    main()
