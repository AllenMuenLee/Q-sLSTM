# scripts/experiments/scalar_tasks/extrapolate_checkpoints.py
#
# Evaluate finished runs' best-validation checkpoints on longer extrapolation sequences (inference only).
# The extrapolation split is regenerated from each run's own data seed, so it is exactly the split a
# --run-extrapolation run at that length would have used. Outputs go to <run>/extrapolation_L<length>/;
# the run's original files are not touched. Each run's test split is also replayed and checked against
# the recorded test MSE, so a mis-loaded checkpoint fails instead of producing numbers.
#
#   python scripts/experiments/scalar_tasks/extrapolate_checkpoints.py \
#       results/scalar_tasks/paper/2026-09-28 results/scalar_tasks/paper/2026-09-30 --length 256

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

TEST_TOLERANCE = 1e-3  # relative; replayed vs recorded test MSE


def evaluate_run(job):
    run_dir, length, save_predictions = job
    import torch

    from q_slstm.datasets.scalar_tasks import generate_dataset
    from q_slstm.experiments import scalar_tasks as st
    from q_slstm.experiments import scalar_tasks_metrics as stm
    from q_slstm.models.factory import match_recurrence

    torch.set_num_threads(1)
    run_dir = Path(run_dir)
    out = run_dir / f"extrapolation_L{length}"
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    config = {**config, "device": "cpu", "batch_size": 256, "run_extrapolation": False}
    gen, seeds, task = st.generation_config(config), config["seeds"], config["task"]
    gen.validate(task, length)

    data = st.make_datasets(config)
    baseline_value = st.training_target_mean(data["train"])
    extrap = generate_dataset(task, config["extrapolation_size"], length, seeds["data"], "extrapolation", gen)

    model = st.build_model(config)
    model.load_state_dict(torch.load(run_dir / "checkpoints" / "best.pt", map_location="cpu",
                                     weights_only=False)["model_state_dict"])
    match_recurrence(model, config.get("qslstm_recurrence"))

    recorded = float(pd.read_csv(run_dir / "per_seed_metrics_test.csv")["mse"].iloc[0])
    t = data["test"].tensors
    pred = st.predict_dataset(model, data["test"], config)
    test_mask = t["loss_mask"][..., 0].numpy().astype(bool)
    replayed = float(((pred - t["targets"][..., 0].numpy()) ** 2)[test_mask].mean())
    if abs(replayed / recorded - 1) > TEST_TOLERANCE:
        raise RuntimeError(f"{run_dir}: replayed test MSE {replayed:.6g} != recorded {recorded:.6g}")

    out.mkdir(exist_ok=True)
    pred = st.predict_dataset(model, extrap, config)
    t = extrap.tensors
    target = t["targets"][..., 0].numpy().astype(np.float64)
    mask = t["loss_mask"][..., 0].numpy().astype(bool)
    run_seed, name, split = seeds["run_seed"], config["model"], "extrapolation"

    per_seq = stm.add_identity(stm.sequence_metrics(pred, target, mask), run_seed, name, task, split,
                               extrap.sequence_ids.numpy())
    per_seed = stm.aggregate_per_seed(per_seq, target, mask, baseline_value)
    per_seq.to_csv(out / "per_sequence_metrics_extrapolation.csv", index=False)
    per_seed.to_csv(out / "per_seed_metrics_extrapolation.csv", index=False)
    abs_err = np.where(mask, np.abs(pred - target), 0.0)
    n_sup = mask.sum(axis=0)
    mae = np.divide(abs_err.sum(axis=0), n_sup, out=np.full(length, np.nan), where=n_sup > 0)
    pd.DataFrame({"timestep": np.arange(length), "mse": stm.timestep_curve(pred, target, mask), "mae": mae}).to_csv(
        out / "timestep_metrics_extrapolation.csv", index=False)
    if save_predictions:
        n = len(pred)
        pd.DataFrame({
            "sequence_id": np.repeat(extrap.sequence_ids.numpy(), length), "timestep": np.tile(np.arange(length), n),
            "input": t["inputs"][..., 0].numpy().ravel(), "target": target.ravel(), "prediction": pred.ravel(),
            "is_supervised": mask.ravel(),
        }).to_csv(out / "predictions_extrapolation.csv", index=False)
    (out / "extrapolation.json").write_text(json.dumps({
        "checkpoint": "checkpoints/best.pt", "train_sequence_length": config["sequence_length"],
        "extrapolation_length": length, "extrapolation_size": config["extrapolation_size"],
        "data_seed": seeds["data"], "recorded_test_mse": recorded, "replayed_test_mse": replayed,
        "baseline_constant_prediction": baseline_value,
    }, indent=2), encoding="utf-8")
    return str(run_dir)


def main(argv=None):
    p = argparse.ArgumentParser(description="Extrapolation evaluation of finished scalar-task runs (no training).")
    p.add_argument("sweeps", nargs="+", type=Path, help="dated sweep folders, e.g. results/scalar_tasks/paper/2026-09-28")
    p.add_argument("--length", type=int, required=True, help="extrapolation sequence length")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--models", type=str, default=None, help="comma-separated models to evaluate (default: all found)")
    p.add_argument("--save-predictions", action="store_true", help="also write predictions_extrapolation.csv (large)")
    p.add_argument("--redo", action="store_true", help="re-evaluate runs that already have results at this length")
    args = p.parse_args(argv)

    runs = sorted(c.parent for s in args.sweeps for c in s.glob("*/seed_*/*/complete.json"))
    if args.models:
        runs = [r for r in runs if r.name in args.models.split(",")]
    if not args.redo:
        runs = [r for r in runs if not (r / f"extrapolation_L{args.length}" / "extrapolation.json").exists()]
    print(f"{len(runs)} runs to evaluate at length {args.length}", flush=True)
    jobs = [(str(r), args.length, args.save_predictions) for r in runs]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for k, _ in enumerate(pool.map(evaluate_run, jobs), 1):
            if k % 50 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
