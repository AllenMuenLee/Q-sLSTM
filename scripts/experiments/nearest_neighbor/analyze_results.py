# scripts/experiments/nearest_neighbor/analyze_results.py
#
# Combine the runs of one sweep into seed tables, the paired QLSTM / Q-sLSTM comparison, plots, and a
# machine-generated Markdown summary. Only metric-step rows (second candidate onward) are used; raw
# input/forget gate values never appear in any input or output of this script.

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from q_slstm.experiments import nearest_neighbor_metrics as nnm  # noqa: E402
from q_slstm.experiments.nearest_neighbor import verify_pairing  # noqa: E402

MODELS = ("qlstm", "qslstm")
MODEL_LABELS = {"qlstm": "QLSTM (conventional)", "qslstm": "Q-sLSTM (stabilized)"}
COLORS = {"qlstm": "#4C72B0", "qslstm": "#DD8452"}
SPLIT_TITLES = {"test": "held-out (trained length)", "extrapolation": "EXTRAPOLATION (length-trained checkpoints)"}
PRIMARY_METRIC = "mse"
COMPARISON_METRICS = ("mse", "mae", "final_mse", "event_mse", "nonevent_mse", "drift_nonevent",
                      "drift_post_best", "revision_gain", "alpha_event", "alpha_nonevent", "alpha_contrast")
LAG_PLOT = nnm.LAGS


# ---------------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------------

def discover_runs(runs_dir):
    """Completed runs under `runs_dir` as {(run_seed, model): (run_dir, config)}."""
    runs, incomplete = {}, []
    for cfg_path in sorted(Path(runs_dir).rglob("config.json")):
        config = json.loads(cfg_path.read_text(encoding="utf-8"))
        if config.get("kind") != "nearest_neighbor_run":
            continue
        run_dir = cfg_path.parent
        if not (run_dir / "complete.json").exists():
            incomplete.append(run_dir)
            continue
        runs[(config["seeds"]["run_seed"], config["model"])] = (run_dir, config)
    labels = {cfg["scale_label"] for _, cfg in runs.values()}
    if len(labels) > 1:
        raise SystemExit(f"runs of several scales found under {runs_dir}: {sorted(labels)}; analyze one at a time")
    return runs, incomplete


def load_table(runs, filename):
    frames = [pd.read_csv(run_dir / filename) for _, (run_dir, _) in sorted(runs.items())
              if (run_dir / filename).exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------------------------

def _finish(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_mse_by_case(per_seed, split, path):
    cases = ["all", *nnm.CASE_ORDER]
    sel = per_seed[per_seed["split"] == split]
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.36
    for k, model in enumerate(MODELS):
        for i, case in enumerate(cases):
            vals = sel[(sel["model"] == model) & (sel["case_type"] == case)].sort_values("run_seed")["mse"].to_numpy()
            x = i + (k - 0.5) * width
            ax.bar(x, vals.mean() if len(vals) else np.nan, width * 0.9, color=COLORS[model], alpha=0.55,
                   label=MODEL_LABELS[model] if i == 0 else None)
            ax.scatter(np.full(len(vals), x), vals, color=COLORS[model], edgecolor="k", s=18, zorder=3)
    ax.set_xticks(range(len(cases)), cases)
    ax.set_ylabel("MSE (second candidate onward)")
    ax.set_title(f"MSE by case type, {SPLIT_TITLES[split]}; points = seeds")
    ax.legend()
    _finish(fig, path)


def _lag_curve(per_seed, split, prefix, case="all"):
    sel = per_seed[(per_seed["split"] == split) & (per_seed["case_type"] == case)]
    lags = [lag for lag in LAG_PLOT if f"{prefix}_{nnm.lag_name(lag)}" in sel.columns]
    return lags, {m: sel[sel["model"] == m].sort_values("run_seed")[[f"{prefix}_{nnm.lag_name(l)}" for l in lags]]
                  .to_numpy(dtype=float) for m in MODELS}


def plot_event_curve(per_seed, split, prefix, ylabel, title, path, cases=("all",)):
    fig, axes = plt.subplots(1, len(cases), figsize=(5.2 * len(cases), 3.8), squeeze=False)
    for ax, case in zip(axes[0], cases):
        lags, curves = _lag_curve(per_seed, split, prefix, case)
        for model in MODELS:
            arr = curves[model]
            if arr.size == 0:
                continue
            for row in arr:
                ax.plot(lags, row, color=COLORS[model], alpha=0.25, lw=0.8)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # lags with no eligible event stay NaN
                mean_curve = np.nanmean(arr, axis=0)
            ax.plot(lags, mean_curve, color=COLORS[model], lw=2, marker="o",
                    label=MODEL_LABELS[model])
        ax.axvline(0, color="k", ls=":", lw=0.8)
        ax.set_xlabel("lag relative to event (steps)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{case}")
    axes[0][0].legend(fontsize=8)
    fig.suptitle(f"{title} - {SPLIT_TITLES[split]}", y=1.02)
    _finish(fig, path)


def plot_alpha_summary(per_seed, split, path):
    sel = per_seed[(per_seed["split"] == split) & (per_seed["case_type"] == "all")]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for k, model in enumerate(MODELS):
        for j, col in enumerate(("alpha_event", "alpha_nonevent")):
            vals = sel[sel["model"] == model][col].dropna().to_numpy()
            x = j + (k - 0.5) * 0.35
            ax.bar(x, vals.mean() if len(vals) else np.nan, 0.32, color=COLORS[model], alpha=0.55,
                   label=MODEL_LABELS[model] if j == 0 else None)
            ax.scatter(np.full(len(vals), x), vals, color=COLORS[model], edgecolor="k", s=18, zorder=3)
    ax.set_xticks([0, 1], ["significant events", "non-event distractors"])
    ax.set_ylabel("write proportion alpha")
    ax.set_title(f"alpha at events vs distractors, {SPLIT_TITLES[split]}")
    ax.legend(fontsize=8)
    _finish(fig, path)


def plot_timestep_curves(curves, split, path, cases=("early", "near_best")):
    fig, axes = plt.subplots(2, len(cases), figsize=(5.2 * len(cases), 6), squeeze=False, sharex=True)
    for j, case in enumerate(cases):
        for i, (col, label) in enumerate((("mse", "squared error"), ("drift", "retention drift |pred_t - pred_(t-1)|"))):
            ax = axes[i][j]
            for model in MODELS:
                sel = curves[(curves["case_type"] == case) & (curves["model"] == model)]
                per_t = sel.groupby("timestep")[col].mean()  # mean over seeds of per-seed timestep means
                ax.plot(per_t.index, per_t.values, color=COLORS[model], lw=2, label=MODEL_LABELS[model])
            ax.set_ylabel(label)
            ax.set_title(case)
        axes[1][j].set_xlabel("timestep (token index)")
    axes[0][0].legend(fontsize=8)
    fig.suptitle(f"Error and retention drift by timestep - {SPLIT_TITLES[split]}", y=1.0)
    _finish(fig, path)


def trace_sequence_ids(preds, per_case=1):
    """Deterministic rule fixed before looking at results: the lowest `per_case` sequence ids per case."""
    ids = {}
    for case, group in preds.groupby("case_type"):
        ids[case] = sorted(group["sequence_id"].unique())[:per_case]
    return ids


def plot_traces(preds, split, path):
    seed = int(preds["run_seed"].min())
    ids = trace_sequence_ids(preds)
    cases = [c for c in nnm.CASE_ORDER if c in ids]
    fig, axes = plt.subplots(len(cases), 1, figsize=(8, 2.6 * len(cases)), squeeze=False, sharex=True)
    for ax, case in zip(axes[:, 0], cases):
        sid = ids[case][0]
        seq = preds[(preds["run_seed"] == seed) & (preds["sequence_id"] == sid)]
        cand = seq[seq["timestep"] >= 1].sort_values("timestep")
        ax.step(cand["timestep"], cand["target"], where="post", color="k", lw=2, label="target")
        ax.plot(cand["timestep"], cand["similarity"], color="gray", ls="--", lw=1, label="similarity")
        for model in MODELS:
            m = seq[(seq["model"] == model) & (seq["timestep"] >= 1)].sort_values("timestep")
            ax.plot(m["timestep"], m["prediction"], color=COLORS[model], lw=1.6, label=MODEL_LABELS[model])
        events = cand[cand["is_event"]]
        ax.scatter(events["timestep"], events["target"], marker="v", color="crimson", zorder=4, label="event")
        ax.set_title(f"{case} (sequence_id {sid}, seed {seed})", fontsize=9)
    axes[0][0].legend(fontsize=7, ncol=3)
    axes[-1][0].set_xlabel("timestep (token index)")
    fig.suptitle(f"Fixed pre-selected traces (lowest id per case) - {SPLIT_TITLES[split]}", y=1.0)
    _finish(fig, path)


# ---------------------------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------------------------

def _fmt(x, digits=4):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{digits}f}"


def _markdown_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.astype(object).iterrows():
        lines.append("| " + " | ".join(_fmt(v) if isinstance(v, (float, np.floating)) else str(v)
                                       for v in row) + " |")
    return "\n".join(lines)


def write_summary(path, runs, cfg, comparisons, pairing, param_table, incomplete, splits):
    seeds = sorted({s for s, _ in runs})
    lines = [
        f"# Nearest-neighbor memory-revision results: scale `{cfg['scale_label']}`",
        "",
        "Machine-generated by `analyze_results.py`; do not edit by hand.",
        "",
        "## Setup",
        f"- Scale preset: `{cfg['scale_preset']}`; preset overrides: `{json.dumps(cfg['preset_overrides'])}`",
        f"- Sequence length: {cfg['sequence_length']} total tokens = 1 reference + {cfg['n_candidates']} candidates",
        f"- Training pool {cfg['train_size']} (optimizer {cfg['optimizer_train_size']}, validation {cfg['val_size']}); "
        f"held-out {cfg['test_size']} (balanced over iid/late/early/near_best)",
        f"- Hidden size {cfg['hidden_size']}, VQC depth {cfg['qnn_depth']}, qubits {cfg['n_qubits']}, "
        f"epochs <= {cfg['epochs']}, patience {cfg['patience']}, batch {cfg['batch_size']}, lr {cfg['lr']}",
        f"- Seeds ({len(seeds)}): {seeds}; models: {sorted({m for _, m in runs})}",
        "- Every metric excludes the reference token and the first candidate (`metric_mask`).",
        "- The best-validation checkpoint was evaluated once on held-out data; no held-out selection.",
        "",
        "## Parameter matching",
        _markdown_table(param_table),
        "",
        "## Pairing verification (same seed, QLSTM vs Q-sLSTM)",
        _markdown_table(pd.DataFrame(pairing).T.reset_index().rename(columns={"index": "run_seed"})
                        .drop(columns=["shared_parameter_names"], errors="ignore")) if pairing else "no paired seeds",
        "",
    ]
    if incomplete:
        lines += ["## Incomplete runs (excluded)", *[f"- `{p}`" for p in incomplete], ""]
    for split in splits:
        title = SPLIT_TITLES[split]
        lines += [f"## {title}", ""]
        prim = comparisons[split]["primary_summary"]
        lines += [
            f"### Primary: {prim['metric']} (all cases, second candidate onward), paired Q-sLSTM - QLSTM",
            f"- Paired seeds: {prim['n_paired_seeds']}; mean difference {_fmt(prim['mean_difference'], 5)} "
            f"(sd {_fmt(prim['sd_difference'], 5)}); 95% CI "
            f"[{_fmt(prim['ci95_low'], 5)}, {_fmt(prim['ci95_high'], 5)}] ({prim['ci_method']})",
            f"- Seeds where Q-sLSTM is lower: {prim['seeds_a_lower']} of {prim['n_paired_seeds']}",
            "",
            _markdown_table(comparisons[split]["primary_table"]),
            "",
            "### Mean and sd across seeds",
            _markdown_table(comparisons[split]["model_summary"][
                ["model", "case_type", "n_seeds", "mse_mean", "mse_sd", "mae_mean", "mae_sd",
                 "final_mse_mean", "event_mse_mean", "nonevent_mse_mean"]]),
            "",
            "### Paired differences (Q-sLSTM - QLSTM) by case and metric",
            _markdown_table(comparisons[split]["paired_summaries"][
                ["case_type", "metric", "n_paired_seeds", "mean_difference", "sd_difference",
                 "ci95_low", "ci95_high", "seeds_a_lower"]]),
            "",
            "### Contributing counts (summed over seeds, all cases)",
            _markdown_table(comparisons[split]["counts"]),
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------------------------

def analyze(runs_dir, out_dir=None):
    runs, incomplete = discover_runs(runs_dir)
    if not runs:
        raise SystemExit(f"no completed nearest-neighbor runs under {runs_dir}")
    missing = [m for m in MODELS if not any(model == m for _, model in runs)]
    if missing:
        raise SystemExit(f"no completed runs for {missing}; run each model with run_sweep.py (same seeds) first")
    out_dir = Path(out_dir) if out_dir else Path(runs_dir) / "analysis"
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    configs = [cfg for _, cfg in runs.values()]

    # ---- pairing + parameter matching ----
    seeds = sorted({s for s, _ in runs})
    unpaired = [s for s in seeds if not all((s, m) in runs for m in MODELS)]
    if unpaired:
        print(f"WARNING: seeds {unpaired} lack a completed run for one model; they are excluded from paired results",
              file=sys.stderr)
    pairing = {}
    for seed in seeds:
        if (seed, "qlstm") in runs and (seed, "qslstm") in runs:
            pairing[seed] = verify_pairing(runs[(seed, "qlstm")][0], runs[(seed, "qslstm")][0])
    (out_dir / "pairing_check.json").write_text(json.dumps(pairing, indent=2), encoding="utf-8")
    rows = []
    for (seed, model), (run_dir, cfg) in sorted(runs.items()):
        params = json.loads((run_dir / "parameters.json").read_text(encoding="utf-8"))
        rows.append({"run_seed": seed, "model": model, "trainable_parameters": params["trainable_parameters"],
                     "n_qubits": params["n_qubits"]})
    param_table = pd.DataFrame(rows)
    param_table["matches_other_model"] = param_table.groupby("run_seed")["trainable_parameters"].transform("nunique") == 1
    param_table.to_csv(out_dir / "parameter_counts.csv", index=False)

    splits = [s for s in ("test", "extrapolation") if any((d / f"per_seed_metrics_{s}.csv").exists()
                                                            for d, _ in runs.values())]
    comparisons = {}
    all_seed_tables = []
    for split in splits:
        per_seed = load_table(runs, f"per_seed_metrics_{split}.csv")
        per_seq = load_table(runs, f"per_sequence_metrics_{split}.csv")
        preds = load_table(runs, f"predictions_{split}.csv")
        all_seed_tables.append(per_seed)
        per_seed.to_csv(out_dir / f"combined_per_seed_metrics_{split}.csv", index=False)
        per_seq.to_csv(out_dir / f"combined_per_sequence_metrics_{split}.csv", index=False)

        table, prim = nnm.paired_difference(per_seed, PRIMARY_METRIC, "all", split)
        table.to_csv(out_dir / f"primary_paired_differences_{split}.csv", index=False)
        summaries = []
        for case in ("all", *nnm.CASE_ORDER):
            for metric in COMPARISON_METRICS:
                if metric in per_seed.columns:
                    summaries.append(nnm.paired_difference(per_seed, metric, case, split)[1])
        paired_summaries = pd.DataFrame(summaries)
        paired_summaries.to_csv(out_dir / f"paired_comparison_{split}.csv", index=False)
        summary = nnm.model_summary(per_seed, list(COMPARISON_METRICS), split)
        summary.to_csv(out_dir / f"model_summary_{split}.csv", index=False)
        counts = (per_seed[(per_seed["split"] == split) & (per_seed["case_type"] == "all")]
                  .groupby("model")[["n_sequences", "n_metric_steps", "n_events", "n_nonevents",
                                     "n_drift_steps", "n_revision_steps"]].sum().reset_index())
        comparisons[split] = {"primary_table": table, "primary_summary": prim, "model_summary": summary,
                              "paired_summaries": paired_summaries, "counts": counts}

        suffix = split
        plot_mse_by_case(per_seed, split, out_dir / "plots" / f"mse_by_case_{suffix}.png")
        plot_event_curve(per_seed, split, "event_abs_err_lag", "absolute error",
                         "Event-aligned absolute error", out_dir / "plots" / f"event_error_curves_{suffix}.png",
                         cases=("all", *nnm.CASE_ORDER[1:]))
        plot_event_curve(per_seed, split, "best_abs_err_lag", "absolute error",
                         "Error around the global-best event", out_dir / "plots" / f"best_event_error_curves_{suffix}.png",
                         cases=("late", "early", "near_best"))
        if "alpha_event" in per_seed.columns:
            plot_event_curve(per_seed, split, "event_alpha_lag", "write proportion alpha",
                             "Event-aligned alpha", out_dir / "plots" / f"event_alpha_curves_{suffix}.png",
                             cases=("all", *nnm.CASE_ORDER[1:]))
            plot_alpha_summary(per_seed, split, out_dir / "plots" / f"alpha_event_vs_distractor_{suffix}.png")
        if not preds.empty:
            curves = nnm.timestep_curves(preds)
            curves.to_csv(out_dir / f"timestep_curves_{split}.csv", index=False)
            plot_timestep_curves(curves, split, out_dir / "plots" / f"timestep_error_drift_{suffix}.png")
            plot_traces(preds, split, out_dir / "plots" / f"traces_{suffix}.png")

    pd.concat(all_seed_tables, ignore_index=True).to_csv(out_dir / "combined_seed_table.csv", index=False)
    write_summary(out_dir / "summary.md", runs, configs[0], comparisons, pairing, param_table, incomplete, splits)
    print(f"analysis written to {out_dir}")
    return out_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description="Analyze a nearest-neighbor sweep.")
    parser.add_argument("--runs-dir", required=True, help="directory containing the run directories")
    parser.add_argument("--out-dir", default=None, help="default: <runs-dir>/analysis")
    args = parser.parse_args(argv)
    analyze(args.runs_dir, args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
