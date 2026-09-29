# solar_next experiment: next-hour Ontario solar generation from the generation history alone
# (1 input, 1 output per hour). Separate from the 13-feature solar_generation experiment.
#
# Data: q_slstm.datasets.solar_next (32-hour windows of the IESO series, chronological split shared
# with solar_generation). Training, evaluation, metrics, and artifacts reuse the scalar-task pipeline
# (sequence-to-sequence, masked MSE at every hour, best-validation checkpoint evaluated once), so the
# models and presets are those of q_slstm.experiments.scalar_tasks.
#
# Run layout: <save_dir>/<scale>/<run date>/seed_<seed>/<model>/

from __future__ import annotations

from pathlib import Path

from q_slstm.datasets.solar_next import TASK, load_solar_next, reference_mse
from q_slstm.experiments import scalar_tasks as st

MODELS = st.MODELS
PRESETS = st.PRESETS
DEFAULT_SAVE_DIR = "results/solar_next"


def add_run_arguments(parser):
    """The scalar-task run arguments without the synthetic-task generation options."""
    st.add_run_arguments(parser, task_options=False, save_dir=DEFAULT_SAVE_DIR)


def resolve_config(args):
    a = dict(vars(args)) if not isinstance(args, dict) else dict(args)
    a["task"] = TASK
    a.setdefault("save_dir", DEFAULT_SAVE_DIR)
    config = st.resolve_config(a, tasks=(TASK,))
    config.pop("generation")
    config["kind"] = "solar_next_run"
    config["data"] = ("IESO hourly solar generation (data/solar_generation), scaled to [-1, 1] by the "
                      "training-period maximum; input = this hour, target = next hour")
    return config


def sweep_directory(config):
    return Path(config["save_dir"]) / config["scale_label"] / config["run_date"]


def run_directory(config):
    return sweep_directory(config) / f"seed_{config['seeds']['run_seed']}" / config["model"]


def make_datasets(config):
    """train / val / test (/ extrapolation); held-out windows are identical for every seed and model."""
    return load_solar_next(config, config["seeds"]["data"])


def dataset_manifest(config, datasets):
    return {
        "task": TASK,
        "seeds": config["seeds"],
        "source": datasets["test"].metadata["source"],
        "splits": {name: {"n_sequences": len(ds), "sequence_length": ds.sequence_length,
                          "warmup_steps": ds.metadata["warmup_steps"], "checksums": ds.checksums()}
                   for name, ds in datasets.items()},
        "reference_mse": {name: reference_mse(ds) for name, ds in datasets.items()},
        "checksum_method": "sha256 over dtype, shape, and little-endian tensor bytes",
    }


def run_experiment(config, run_dir=None):
    return st.run_experiment(config, run_dir or run_directory(config),
                             datasets_fn=make_datasets, manifest_fn=dataset_manifest)
