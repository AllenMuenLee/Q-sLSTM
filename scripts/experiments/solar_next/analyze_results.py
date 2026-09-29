# scripts/experiments/solar_next/analyze_results.py
#
# Summarize one solar_next sweep folder (completed runs only):
#   summary.csv / summary.md   test MSE, MAE, RMSE per model (mean and sample sd over seeds), in scaled
#                              units and in MW, next to the persistence, daily-persistence, and constant
#                              references; paired differences against QLSTM (same seeds)
#   error_by_model.png         test RMSE (MW) per seed and model with the reference lines
#   figures/                   training curves, test MSE by hour in the window, input/forget gates
#                              (scripts/experiments/scalar_tasks/plot_curves_and_gates.py)
#
# References are computed on the same test windows and steps as the models:
#   persistence        next hour = this hour
#   daily persistence  next hour = same hour yesterday (only steps with that hour inside the window)
#   constant           the training-target mean

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "experiments" / "scalar_tasks"))

import plot_curves_and_gates as curves  # noqa: E402

MODELS = ("qlstm", "qslstm", "qslstm_log", "fk_qslstm", "fk_qlstm", "lstm")
LABELS = {**curves.LABELS, "lstm": "LSTM (classical)"}
COLORS = {**curves.COLORS, "lstm": "#8C8C8C"}
SURFACE, INK, MUTED = curves.SURFACE, curves.INK, curves.MUTED
DAY = 24


def discover(runs_dir):
    runs = {}
    for complete in sorted(Path(runs_dir).glob("seed_*/*/complete.json")):
        run_dir = complete.parent
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        if config.get("kind") == "solar_next_run":
            runs[(config["seeds"]["run_seed"], config["model"])] = run_dir
    return runs


def reference_errors(predictions):
    """Persistence / daily persistence / constant errors on the scored test steps of one run."""
    p = predictions[predictions["is_supervised"]].sort_values(["sequence_id", "timestep"])
    err = {"persistence": p["input"] - p["target"]}
    lagged = predictions.sort_values(["sequence_id", "timestep"]).groupby("sequence_id")["input"].shift(DAY - 1)
    daily = (lagged - predictions["target"]).loc[p.index].dropna()
    if len(daily):  # windows shorter than a day have no same-hour-yesterday input
        err["daily persistence"] = daily
    return {name: {"mse": float((e ** 2).mean()), "mae": float(e.abs().mean()), "n_steps": int(len(e))}
            for name, e in err.items()}


def paired(per_seed, model, metric):
    wide = per_seed.pivot(index="run_seed", columns="model", values=metric)[["qlstm", model]].dropna()
    diff = wide[model] - wide["qlstm"]
    p = stats.wilcoxon(diff).pvalue if len(diff) >= 6 and diff.abs().sum() > 0 else float("nan")
    return {"model": model, "metric": metric, "n_paired_seeds": len(diff), "mean_difference": diff.mean(),
            "relative_difference": diff.mean() / wide["qlstm"].mean(), "wins_vs_qlstm": int((diff < 0).sum()),
            "wilcoxon_p": p}


def plot_rmse(per_seed, refs, scale_mw, models, path):
    fig, ax = plt.subplots(figsize=(1.6 * len(models) + 3.5, 4.2), facecolor=SURFACE)
    curves.style_axes(ax)
    rng = np.random.default_rng(0)
    for k, model in enumerate(models):
        values = per_seed.loc[per_seed["model"] == model, "rmse_mw"].to_numpy()
        ax.scatter(k + rng.uniform(-0.15, 0.15, len(values)), values, s=24, color=COLORS[model],
                   edgecolor=SURFACE, linewidth=1, zorder=3)
        ax.errorbar(k + 0.3, values.mean(), yerr=values.std(ddof=1) if len(values) > 1 else 0, fmt="o",
                    color=INK, capsize=4, zorder=4)
    for (name, ref), style in zip(refs.items(), ((0, (4, 3)), (0, (1, 2)), "-")):
        rmse = np.sqrt(ref["mse"]) * scale_mw / 2
        ax.axhline(rmse, color=MUTED, lw=1.2, ls=style, zorder=1)
        ax.text(len(models) - 0.45, rmse, f"{name} {rmse:.1f}", color=MUTED, fontsize=8, va="bottom", ha="right")
    ax.set_xticks(range(len(models)), [LABELS[m] for m in models], fontsize=8, color=MUTED)
    ax.set_ylabel("test RMSE (MW)", color=MUTED, fontsize=9)
    ax.set_title("Next-hour solar generation: test RMSE per seed (dot + bar = mean ± sd)",
                 color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Summarize one solar_next sweep.")
    parser.add_argument("--runs-dir", type=Path, required=True, help="e.g. results/solar_next/paper/2026-09-30")
    parser.add_argument("--no-curves", action="store_true", help="skip the training-curve and gate figures")
    parser.add_argument("--workers", type=int, default=4, help="processes for the gate replay")
    args = parser.parse_args(argv)

    runs = discover(args.runs_dir)
    if not runs:
        raise SystemExit(f"no completed solar_next runs under {args.runs_dir}")
    models = [m for m in MODELS if any(m == model for _, model in runs)]
    manifest = json.loads((next(iter(runs.values())) / "dataset_manifest.json").read_text(encoding="utf-8"))
    scale_mw = manifest["source"]["scale_mw"]
    to_mw2 = (scale_mw / 2) ** 2

    per_seed = pd.concat([pd.read_csv(d / "per_seed_metrics_test.csv") for d in runs.values()], ignore_index=True)
    per_seed["rmse"] = np.sqrt(per_seed["mse"])
    per_seed["mse_mw2"], per_seed["mae_mw"], per_seed["rmse_mw"] = (
        per_seed["mse"] * to_mw2, per_seed["mae"] * scale_mw / 2, per_seed["rmse"] * scale_mw / 2)

    refs = reference_errors(pd.read_csv(next(iter(runs.values())) / "predictions_test.csv"))
    refs["constant"] = {"mse": float(per_seed["baseline_mse"].iloc[0]), "mae": float("nan"), "n_steps": None}
    per_seed["skill_vs_persistence"] = 1 - per_seed["mse"] / refs["persistence"]["mse"]

    metrics = ["mse", "mae", "rmse", "mse_mw2", "mae_mw", "rmse_mw", "r2", "skill_vs_persistence"]
    summary = per_seed.groupby("model")[metrics].agg(["mean", "std"])
    summary.columns = [f"{a}_{'sd' if b == 'std' else b}" for a, b in summary.columns]
    summary.insert(0, "n_seeds", per_seed.groupby("model")["run_seed"].nunique())
    summary = summary.loc[models].reset_index()
    ref_rows = pd.DataFrame([{"model": name, "n_seeds": None, "mse_mean": r["mse"], "mae_mean": r["mae"],
                              "rmse_mean": np.sqrt(r["mse"]), "mse_mw2_mean": r["mse"] * to_mw2,
                              "mae_mw_mean": r["mae"] * scale_mw / 2, "rmse_mw_mean": np.sqrt(r["mse"]) * scale_mw / 2,
                              "skill_vs_persistence_mean": 1 - r["mse"] / refs["persistence"]["mse"]}
                             for name, r in refs.items()])
    table = pd.concat([summary, ref_rows], ignore_index=True)
    table.to_csv(args.runs_dir / "summary.csv", index=False)
    comparisons = pd.DataFrame([paired(per_seed, m, metric) for m in models if m != "qlstm"
                                for metric in ("mse", "mae", "rmse")]) if "qlstm" in models else pd.DataFrame()
    comparisons.to_csv(args.runs_dir / "paired_vs_qlstm.csv", index=False)
    plot_rmse(per_seed, refs, scale_mw, models, args.runs_dir / "error_by_model.png")

    fmt = lambda m, s: f"{m:.2f} ± {s:.2f}" if pd.notna(s) else f"{m:.2f}"
    lines = [
        f"# solar_next: next-hour Ontario solar generation from generation alone ({args.runs_dir.name})", "",
        f"Test windows: {manifest['splits']['test']['n_sequences']} x {manifest['splits']['test']['sequence_length']} h "
        f"from {manifest['source']['split_boundaries']['test']['first_timestamp_utc']} to "
        f"{manifest['source']['split_boundaries']['test']['last_timestamp_utc']}; errors in MW use scale "
        f"{scale_mw:.0f} MW. Mean ± sample sd over seeds.", "",
        "| model | seeds | MSE (MW²) | MAE (MW) | RMSE (MW) | R² | skill vs persistence |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in table.iterrows():
        name = LABELS.get(r["model"], r["model"])
        lines.append(f"| {name} | {'' if pd.isna(r['n_seeds']) else int(r['n_seeds'])} | "
                     f"{fmt(r['mse_mw2_mean'], r.get('mse_mw2_sd'))} | "
                     f"{'' if pd.isna(r['mae_mw_mean']) else fmt(r['mae_mw_mean'], r.get('mae_mw_sd'))} | "
                     f"{fmt(r['rmse_mw_mean'], r.get('rmse_mw_sd'))} | "
                     f"{'' if pd.isna(r.get('r2_mean')) else fmt(r['r2_mean'], r.get('r2_sd'))} | "
                     f"{r['skill_vs_persistence_mean']:+.3f} |")
    if len(comparisons):
        lines += ["", "## Paired against QLSTM (same seeds; negative difference = lower error than QLSTM)", "",
                  "| model | metric | seeds | mean difference (scaled) | relative | wins | Wilcoxon p |",
                  "|---|---|---|---|---|---|---|"]
        for _, c in comparisons.iterrows():
            lines.append(f"| {LABELS[c['model']]} | {c['metric'].upper()} | {c['n_paired_seeds']} | "
                         f"{c['mean_difference']:+.5f} | {c['relative_difference']:+.1%} | "
                         f"{c['wins_vs_qlstm']}/{c['n_paired_seeds']} | {c['wilcoxon_p']:.3g} |")
    lines += ["", "References use the same test steps; daily persistence only scores steps whose same hour "
                  "yesterday lies inside the window. Parameter counts differ by model and are not matched."]
    (args.runs_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.stdout.reconfigure(errors="replace")  # consoles without UTF-8 (e.g. cp950) cannot show ² or ±
    print("\n".join(lines))

    if not args.no_curves:
        curves.main(["--runs-dir", str(args.runs_dir), "--workers", str(args.workers)])
    return 0


if __name__ == "__main__":
    sys.exit(main())
