# scripts/experiments/solar_generation/train_solar_generation.py
#
# Train and evaluate one model (conventional QLSTM or stabilized Q-sLSTM) on one seed of the solar
# generation forecasting task. Use run_sweep.py to run a model over paired seeds.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(ROOT / "src"))

import pandas  # noqa: E402,F401  (import before torch: avoids a Windows pyarrow/torch DLL crash)

from q_slstm.models.factory import QUANTUM_MODELS  # noqa: E402
from q_slstm.experiments.solar_generation import add_run_arguments, resolve_config, run_directory, run_experiment  # noqa: E402


def build_parser():
    parser = argparse.ArgumentParser(description="Solar-generation forecasting experiment (one run).")
    parser.add_argument("--model", choices=QUANTUM_MODELS, required=True)
    parser.add_argument("--seed", type=int, default=0, help="run seed; paired models share it")
    parser.add_argument("--resume", action="store_true",
                        help="skip a verified complete run, or continue an interrupted one from last.pt")
    parser.add_argument("--overwrite", action="store_true", help="rerun even if a complete run exists")
    add_run_arguments(parser)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = resolve_config(args)
    s = config["study"]
    print(f"study={config['study_id']} scale={s['scale_label']} model={config['model']} "
          f"seed={config['seeds']['run_seed']} dataset={s['dataset_id']} L={s['sequence_length']} "
          f"projection 13->{s['projection_size']} qubits={s['n_qubits']} depth={s['qnn_depth']} "
          f"epochs={s['epochs']} batch={s['batch_size']}", flush=True)
    print(f"run directory: {run_directory(config)}", flush=True)
    run_dir = run_experiment(config, resume=args.resume, overwrite=args.overwrite)
    print(f"run complete: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
