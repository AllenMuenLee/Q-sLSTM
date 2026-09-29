"""Replay the exact sources of a comparison and plot unscaled input or forget gates by gap.

Uses all test sequences and the same eligible steps/bins as plot_key_results.
Hooks observe the real forward pass; saved predictions validate checkpoint replay.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from analyze_results import COLORS
from compare_variants import VARIANTS
from gate_trace import load_model
from plot_key_results import SIGNED_BINS
from q_slstm.experiments.nearest_neighbor import make_datasets
from q_slstm.models.q_slstm_cell import QSLSTM_RECURRENCE, bounded_log_ratio
from q_slstm.models.q_slstm_log_cell import QSLSTM_LOG_RECURRENCE, logarithmic_gate

GATES = ("input", "forget")

LABELS = {
    "qlstm": "QLSTM: sigmoid(q)",
    "qslstm": "QsLSTM: (1+q)/(1-q), before stabilization",
    "qslstm_log": "QsLSTM_LOG: ln(2/(1-q)), before scaling",
}
SHORT_NAMES = {"qlstm": "QLSTM", "qslstm": "QsLSTM", "qslstm_log": "QsLSTM_LOG"}
SIGMOID_LABELS = {
    "qslstm": "QsLSTM: sigmoid(q)",
    "qslstm_log": "QsLSTM_LOG: sigmoid(q)",
}


def amplified(config, gate):
    """Input gates are always amplified; forget gates only in runs predating the sigmoid-forget recurrences."""
    return gate == "input" or config.get("qslstm_recurrence") not in (QSLSTM_RECURRENCE, QSLSTM_LOG_RECURRENCE)


def gate_value(q, config, gate):
    if config["model"] == "qlstm" or not amplified(config, gate):
        return torch.sigmoid(q)
    if config["model"] == "qslstm":
        return torch.exp(bounded_log_ratio(q, config["gate_epsilon"]))
    return logarithmic_gate(q)


def trace(run_dir, config, batch_size, gate):
    ds = make_datasets({**config, "run_extrapolation": False})["test"]
    model = load_model(run_dir, config)
    captured = []

    def observe(_module, _inputs, q):
        captured.append(gate_value(q, config, gate).detach().double().mean(dim=-1).cpu())

    handle = getattr(model.cell, f"{gate}_gate").register_forward_hook(observe)
    gates, predictions = [], []
    try:
        with torch.no_grad():
            for start in range(0, len(ds), batch_size):
                captured.clear()
                x = ds.tensors["inputs"][start:start + batch_size]
                pred, _ = model(x)
                assert len(captured) == x.shape[1]
                gates.append(torch.stack(captured, dim=1).numpy())
                predictions.append(pred[..., 0].cpu().numpy())
    finally:
        handle.remove()
    gates, predictions = np.concatenate(gates), np.concatenate(predictions)
    n, length = predictions.shape
    replay = pd.DataFrame({
        "sequence_id": np.repeat(ds.sequence_ids.numpy(), length),
        "timestep": np.tile(np.arange(length), n),
        "gate": gates.ravel(), "replayed_prediction": predictions.ravel(),
        "replayed_target": ds.tensors["targets"].numpy().reshape(-1),
    })
    saved = pd.read_csv(run_dir / "predictions_test.csv").sort_values(["sequence_id", "timestep"])
    group = saved.groupby("sequence_id")
    saved["signed_gap"] = saved["similarity"] - group["running_best_similarity"].shift()
    previous_metric = group["is_metric_step"].shift().eq(True)
    saved["eligible"] = saved["is_metric_step"] & previous_metric
    steps = saved.merge(replay, on=["sequence_id", "timestep"], validate="one_to_one")
    assert len(steps) == n * length
    candidates = steps["timestep"] > 0  # Saved reference-token targets are deliberately NaN.
    np.testing.assert_allclose(steps.loc[candidates, "target"], steps.loc[candidates, "replayed_target"],
                               atol=1e-7, rtol=0)
    np.testing.assert_allclose(steps["prediction"], steps["replayed_prediction"], atol=2e-5, rtol=0)
    max_error = float((steps["prediction"] - steps["replayed_prediction"]).abs().max())
    steps = steps[steps["eligible"]].copy()
    assert np.isfinite(steps["gate"]).all()
    steps["bin"] = pd.cut(steps["signed_gap"], SIGNED_BINS, include_lowest=True)
    assert steps["bin"].notna().all()
    bins = steps.groupby("bin", observed=True).agg(
        value=("gate", "mean"), gap=("signed_gap", "mean"), n_steps=("gate", "size")
    ).reset_index()
    bins["bin"] = bins["bin"].astype(str)
    bins["model"], bins["run_seed"] = config["model"], config["seeds"]["run_seed"]
    bins["prediction_max_abs_difference"] = max_error
    bins["amplified"] = config["model"] != "qlstm" and amplified(config, gate)
    return bins


def trace_source(key, source, batch_size, threads, gate):
    torch.set_num_threads(threads)
    start = time.perf_counter()
    run_dir = ROOT / source
    config = json.loads((run_dir / "config.json").read_text())
    assert key == f"{config['seeds']['run_seed']}/{config['model']}"
    return key, trace(run_dir, config, batch_size, gate), time.perf_counter() - start


def model_label(summary, model):
    rows = summary[summary["model"] == model]
    if model in SIGMOID_LABELS and "amplified" in rows and not rows["amplified"].any():
        return SIGMOID_LABELS[model]
    return LABELS[model]


def plot(summary, path, gate="input"):
    name = f"{gate.capitalize()} gate"
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.axvspan(0, 2, color="0.93", zorder=0)
    ax.axvline(0, color="black", lw=0.8, ls=":")
    for model in VARIANTS:
        rows = summary[summary["model"] == model].sort_values("gap")
        ax.plot(rows["gap"], rows["mean"], marker="o", lw=2, color=COLORS[model], label=model_label(summary, model))
        ax.fill_between(rows["gap"], rows["mean"] - rows["sd"], rows["mean"] + rows["sd"],
                        color=COLORS[model], alpha=0.15)
    ax.set_xscale("symlog", linthresh=0.02)
    ax.set_xlim(-2, 2)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("current similarity - running best before the step (symlog)")
    ax.set_ylabel(f"{name} value (before numerical rescaling)")
    ax.set_title(f"{name} vs similarity gap\nMean over hidden units and steps within each seed; band = +/-1 SD across seeds")
    ax.text(0.02, 0.97, "Not a record (gap < 0)", transform=ax.transAxes, va="top", color="0.3")
    ax.text(0.98, 0.97, "New record (gap > 0)", transform=ax.transAxes, va="top", ha="right", color="0.3")
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)

    # Native gate scales differ substantially; individual axes reveal each curve's shape.
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.7), sharex=True)
    for ax, model in zip(axes, VARIANTS):
        rows = summary[summary["model"] == model].sort_values("gap")
        ax.axvspan(0, 2, color="0.93", zorder=0)
        ax.axvline(0, color="black", lw=0.8, ls=":")
        ax.plot(rows["gap"], rows["mean"], marker="o", lw=2, color=COLORS[model])
        ax.fill_between(rows["gap"], rows["mean"] - rows["sd"], rows["mean"] + rows["sd"],
                        color=COLORS[model], alpha=0.15)
        ax.set_xscale("symlog", linthresh=0.02)
        ax.set_xlim(-2, 2)
        ax.set_title(model_label(summary, model).replace(": ", "\n").replace(", before", "\nbefore"), fontsize=10)
        ax.set_xlabel("current similarity - prior running best")
        ax.set_ylabel(f"{name} value")
        ax.grid(alpha=0.3)
    fig.suptitle(f"{name} vs similarity gap (separate y-scales)\nMean +/-1 SD across seeds; shaded right side = new records")
    fig.tight_layout()
    fig.savefig(path.with_name(f"{gate}_gate_vs_gap_by_model.png"), dpi=180)
    plt.close(fig)


def plot_combined(out):
    """Input and forget gates per model from the per-seed CSVs; median and interquartile range over seeds,
    since single seeds with saturated q dominate the QsLSTM mean and SD."""
    per_seed = {gate: pd.read_csv(out / f"{gate}_gate_vs_gap_per_seed.csv") for gate in GATES}
    styles = {"input": dict(ls="-", marker="o"), "forget": dict(ls="--", marker="s", mfc="white")}
    fig, axes = plt.subplots(1, len(VARIANTS), figsize=(14, 4.9), sharex=True)
    for ax, model in zip(axes, VARIANTS):
        ax.axvspan(0, 2, color="0.93", zorder=0)
        ax.axvline(0, color="black", lw=0.8, ls=":")
        labels = []
        for gate in GATES:
            rows = per_seed[gate][per_seed[gate]["model"] == model]
            summary = rows.groupby("bin", observed=True).agg(
                gap=("gap", "mean"), median=("value", "median"),
                q1=("value", lambda v: v.quantile(0.25)), q3=("value", lambda v: v.quantile(0.75)),
            ).sort_values("gap")
            ax.plot(summary["gap"], summary["median"], lw=2, color=COLORS[model], label=f"{gate} gate",
                    **styles[gate])
            ax.fill_between(summary["gap"], summary["q1"], summary["q3"], color=COLORS[model],
                            alpha=0.18 if gate == "input" else 0.08, hatch=None if gate == "input" else "//")
            labels.append(f"{gate}: {model_label(rows, model).split(': ')[1]}")
        ax.set_xscale("symlog", linthresh=0.02)
        ax.set_xlim(-2, 2)
        ax.set_title(f"{SHORT_NAMES[model]}\n" + "\n".join(labels), fontsize=9)
        ax.set_xlabel("current similarity - prior running best")
        ax.set_ylabel("gate value (before numerical rescaling)")
        ax.legend(fontsize=8, loc="center right")
        ax.grid(alpha=0.3)
    fig.suptitle("Input and forget gates vs similarity gap (separate y-scales)\n"
                 "Median across seeds of each seed's mean; band = interquartile range; shaded right side = new records")
    fig.tight_layout()
    fig.savefig(out / "gates_vs_gap.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--gate", choices=GATES, default="input")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--plot-only", action="store_true", help="Redraw figures from the completed summary CSV")
    parser.add_argument("--combine", action="store_true",
                        help="Draw input and forget gates together from both completed per-seed CSVs")
    args = parser.parse_args()
    if args.combine:
        plot_combined(args.analysis_dir / "key_plots")
        return
    stem = f"{args.gate}_gate_vs_gap"
    if args.plot_only:
        out = args.analysis_dir / "key_plots"
        plot(pd.read_csv(out / f"{stem}.csv"), out / f"{stem}.png", args.gate)
        return
    torch.set_num_threads(args.threads)
    sources = json.loads((args.analysis_dir / "sources.json").read_text())
    keys = {(int(key.split("/")[0]), key.split("/")[1]) for key in sources}
    seeds = {seed for seed, _ in keys}
    assert keys == {(seed, model) for seed in seeds for model in VARIANTS}
    out = args.analysis_dir / "key_plots"
    out.mkdir(parents=True, exist_ok=True)
    frames = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(trace_source, key, source, args.batch_size, args.threads, args.gate)
                   for key, source in sources.items()]
        for count, future in enumerate(as_completed(futures), 1):
            key, frame, elapsed = future.result()
            frames.append(frame)
            pd.concat(frames, ignore_index=True).to_csv(out / f"{stem}_per_seed.csv", index=False)
            print(f"[{count}/{len(sources)}] {key}: {elapsed:.1f}s; "
                  f"max prediction difference={frame['prediction_max_abs_difference'].max():.2e}", flush=True)
    per_seed = pd.concat(frames, ignore_index=True)
    summary = per_seed.groupby(["model", "bin"], observed=True).agg(
        mean=("value", "mean"), sd=("value", "std"), gap=("gap", "mean"),
        n_seeds=("value", "size"), steps_per_seed=("n_steps", "mean"), amplified=("amplified", "all")
    ).reset_index().sort_values(["model", "gap"])
    # Identical bin membership and seed coverage to the existing revision plot.
    expected = pd.read_csv(out / "revision_gain_vs_gap.csv").sort_values(["model", "gap"])
    pd.testing.assert_frame_equal(summary[["model", "bin", "n_seeds", "steps_per_seed"]].reset_index(drop=True),
                                  expected[["model", "bin", "n_seeds", "steps_per_seed"]].reset_index(drop=True))
    summary.to_csv(out / f"{stem}.csv", index=False)
    plot(summary, out / f"{stem}.png", args.gate)
    notes = (
        f"# {args.gate.capitalize()} gate vs signed similarity gap\n\n"
        f"All {len(seeds)} paired seeds, all test sequences, best-validation checkpoints from ../sources.json.\n\n"
        "Gap = current similarity minus the running best BEFORE the current step. "
        "Eligibility and bins match revision_gain_vs_gap.csv: both this step and its predecessor must be metric steps.\n\n"
        f"The plotted quantity is the transformed {args.gate} gate before numerical stabilization/scaling, "
        "not the raw circuit expectation q and not the effective (stabilized/scaled) weight the recurrence applies. "
        "QLSTM: sigmoid(q); amplified QsLSTM: exp(log(1+q)-log(1-q)) with the saved epsilon clamp; "
        "amplified QsLSTM_LOG: ln(2/(1-q)); runs with a sigmoid-forget recurrence: sigmoid(q). "
        f"Amplified per model: {per_seed.groupby('model')['amplified'].all().to_dict()}. "
        "Different gate scales do not directly imply different realized memory writes.\n\n"
        "Average over hidden units per step, then eligible steps per bin within each seed. "
        "Lines average these seed means; bands show sample standard deviation across seeds.\n\n"
        f"Checkpoint replay used the actual model forward pass with an observation-only {args.gate}-gate hook. "
        "All regenerated targets and saved predictions were checked. "
        f"Maximum prediction difference: {per_seed['prediction_max_abs_difference'].max():.9g} "
        "(absolute tolerance 2e-5). Bin counts match the existing revision plot.\n"
    )
    (out / f"{stem}.md").write_text(notes, encoding="utf-8")
    print(f"Written to {out}", flush=True)


if __name__ == "__main__":
    main()
