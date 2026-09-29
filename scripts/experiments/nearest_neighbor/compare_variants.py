# scripts/experiments/nearest_neighbor/compare_variants.py
#
# Side-by-side comparison of QLSTM, Q-sLSTM and Q-sLSTM-log drawn from several sweeps (e.g. the
# QLSTM vs Q-sLSTM sweep and the QLSTM vs Q-sLSTM-log sweep). Runs are paired by run seed; a model
# present in more than one sweep (QLSTM) must have identical per-seed metrics in each, otherwise the
# sweeps are not comparable and the script stops.
#
# Every pair of models gets the paired (a - b) Student-t interval from nearest_neighbor_metrics plus a
# Wilcoxon signed-rank p-value; seeds, not sequences or timesteps, are the samples.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

import analyze_results as ar  # noqa: E402
from analyze_results import COLORS, COMPARISON_METRICS, MODEL_LABELS, SPLIT_TITLES  # noqa: E402
from q_slstm.experiments import nearest_neighbor_metrics as nnm  # noqa: E402

ALL_VARIANTS = ("qlstm", "qslstm", "qslstm_log", "fk_qslstm", "fk_qlstm")
VARIANTS = ALL_VARIANTS
# (a, b): differences are a - b, so negative = a has the lower value.
ALL_PAIRS = (("qslstm", "qlstm"), ("qslstm_log", "qlstm"), ("qslstm_log", "qslstm"),
             ("fk_qslstm", "qlstm"), ("fk_qslstm", "qslstm"),
             ("fk_qlstm", "qlstm"), ("fk_qslstm", "fk_qlstm"))  # last: gate design with encoders fixed
PAIRS = ALL_PAIRS


def use_variants(variants):
    """Restrict the comparison to `variants` (e.g. older sweeps that predate the fk models)."""
    global VARIANTS, PAIRS
    VARIANTS = tuple(m for m in ALL_VARIANTS if m in variants)
    PAIRS = tuple((a, b) for a, b in ALL_PAIRS if a in VARIANTS and b in VARIANTS)
SHORT = {"qlstm": "QLSTM", "qslstm": "Q-sLSTM", "qslstm_log": "Q-sLSTM-log", "fk_qslstm": "Q-sLSTM (fk)",
         "fk_qlstm": "QLSTM (fk)"}
CASES = ("all", *nnm.CASE_ORDER)
PANEL_METRICS = {
    "mse": "MSE", "mae": "MAE", "final_mse": "final-step MSE", "event_mse": "event-step MSE",
    "nonevent_mse": "non-event MSE", "drift_nonevent": "non-event drift",
    "revision_gain": "revision gain (higher = better)", "alpha_contrast": "alpha contrast (event - non-event)",
}
OVERVIEW_METRICS = {"revision_gain": "revision gain", "best_epoch": "best epoch", "mse": "MSE", "mae": "MAE",
                    "rmse": "RMSE"}


# ---------------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------------

def collect_runs(runs_dirs):
    """{(run_seed, model): (run_dir, config)} over several sweeps; shared models must agree exactly."""
    runs = {}
    for runs_dir in runs_dirs:
        found, incomplete = ar.discover_runs(runs_dir, models=VARIANTS)
        if incomplete:
            print(f"warning: {len(incomplete)} incomplete run(s) under {runs_dir} ignored", file=sys.stderr)
        for key, value in found.items():
            if key in runs:
                _require_same_run(runs[key][0], value[0])
                continue
            runs[key] = value
    missing = [m for m in VARIANTS if not any(model == m for _, model in runs)]
    if missing:
        raise SystemExit(f"no completed runs of {missing} under {runs_dirs}")
    return runs


def _require_same_run(dir_a, dir_b):
    read = lambda p: pd.read_csv(p).drop(columns=["epoch_seconds"], errors="ignore")  # wall clock always differs
    for name in ("per_seed_metrics_test.csv", "per_seed_metrics_extrapolation.csv", "history.csv"):
        a, b = dir_a / name, dir_b / name
        if a.exists() != b.exists() or (a.exists() and not read(a).equals(read(b))):
            raise SystemExit(f"{dir_a} and {dir_b} are the same seed/model but {name} differs; "
                             "the sweeps are not comparable")


def load_history(runs):
    frames = []
    for (seed, model), (run_dir, _) in sorted(runs.items()):
        h = pd.read_csv(run_dir / "history.csv")
        h["run_seed"], h["model"] = seed, model
        frames.append(h)
    return pd.concat(frames, ignore_index=True)


def paired_seeds(per_seed):
    """Seeds with a completed run of every variant; all comparisons use this common set."""
    sel = per_seed[per_seed["case_type"] == "all"]
    return sorted(set.intersection(*(set(sel[sel["model"] == m]["run_seed"]) for m in VARIANTS)))


# ---------------------------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------------------------

def pairwise_table(per_seed, split, metrics):
    rows = []
    for a, b in PAIRS:
        for case in CASES:
            for metric in metrics:
                if metric not in per_seed.columns:
                    continue
                table, summary = nnm.paired_difference(per_seed, metric, case, split, model_a=a, model_b=b)
                d = table["difference"].dropna().to_numpy(dtype=float)
                summary["wilcoxon_p"] = (stats.wilcoxon(d).pvalue if len(d) > 1 and np.any(d != 0)
                                         else float("nan"))
                summary["mean_a"], summary["mean_b"] = table[a].mean(), table[b].mean()
                summary["relative_difference"] = summary["mean_difference"] / summary["mean_b"]
                rows.append(summary)
    return pd.DataFrame(rows)


def overview_table(per_seed, val_table, split):
    """Mean and sd over seeds per model and case: revision gain, MSE, MAE, RMSE, best epoch.

    RMSE is sqrt of each seed's MSE, then averaged over seeds. Best epoch is per run, so it is the
    same for every case type of a model.
    """
    sel = per_seed[per_seed["split"] == split].copy().assign(rmse=lambda d: np.sqrt(d["mse"]))
    sel = sel.merge(val_table[["run_seed", "model", "best_epoch"]], on=["run_seed", "model"])
    rows = []
    for model in VARIANTS:
        for case in CASES:
            g = sel[(sel["model"] == model) & (sel["case_type"] == case)]
            row = {"split": split, "model": model, "case_type": case, "n_seeds": len(g)}
            for metric in OVERVIEW_METRICS:
                row[f"{metric}_mean"], row[f"{metric}_sd"] = g[metric].mean(), g[metric].std(ddof=1)
            rows.append(row)
    return pd.DataFrame(rows)


def validation_table(hist):
    best = hist.loc[hist.groupby(["run_seed", "model"])["val_mse"].idxmin(),
                    ["run_seed", "model", "epoch", "train_loss", "val_mse"]]
    return best.rename(columns={"epoch": "best_epoch", "train_loss": "train_loss_at_best",
                                "val_mse": "best_val_mse"}).reset_index(drop=True)


# ---------------------------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------------------------

def _finish(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_metric_by_case(per_seed, split, metric, path):
    sel = per_seed[per_seed["split"] == split]
    fig, ax = plt.subplots(figsize=(9, 4))
    width = 0.8 / len(VARIANTS)
    for k, model in enumerate(VARIANTS):
        for i, case in enumerate(CASES):
            vals = sel[(sel["model"] == model) & (sel["case_type"] == case)][metric].dropna().to_numpy()
            x = i + (k - (len(VARIANTS) - 1) / 2) * width
            ax.bar(x, vals.mean() if len(vals) else np.nan, width * 0.9, color=COLORS[model], alpha=0.55,
                   label=MODEL_LABELS[model] if i == 0 else None)
            ax.scatter(np.full(len(vals), x), vals, color=COLORS[model], edgecolor="k", s=12, zorder=3)
    ax.set_xticks(range(len(CASES)), CASES)
    ax.set_ylabel(PANEL_METRICS.get(metric, metric))
    ax.set_title(f"{PANEL_METRICS.get(metric, metric)} by case type, {SPLIT_TITLES[split]}; points = seeds")
    ax.legend(fontsize=8)
    _finish(fig, path)


def plot_seed_lines(per_seed, split, path, case="all"):
    """One panel per metric; x = model, one grey line per seed joins its three runs."""
    sel = per_seed[(per_seed["split"] == split) & (per_seed["case_type"] == case)]
    metrics = [m for m in PANEL_METRICS if m in sel.columns]
    ncols = 4
    nrows = -(-len(metrics) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.2 * nrows), squeeze=False)
    x = np.arange(len(VARIANTS))
    for ax, metric in zip(axes.flat, metrics):
        wide = sel.pivot(index="run_seed", columns="model", values=metric)[list(VARIANTS)]
        for _, row in wide.iterrows():
            ax.plot(x, row.to_numpy(), color="0.6", lw=0.7, alpha=0.7)
        for k, model in enumerate(VARIANTS):
            ax.scatter(np.full(len(wide), k), wide[model], color=COLORS[model], s=14, zorder=3)
            ax.errorbar(k, wide[model].mean(), yerr=wide[model].std(ddof=1), color="k", capsize=4,
                        marker="D", ms=5, zorder=4)
        ax.set_xticks(x, [SHORT[m] for m in VARIANTS], fontsize=8)
        ax.set_title(PANEL_METRICS[metric], fontsize=9)
    for ax in axes.flat[len(metrics):]:
        ax.axis("off")
    fig.suptitle(f"Per-seed comparison ({case}), {SPLIT_TITLES[split]}; lines join one seed, "
                 f"diamond = mean ± sd", y=1.0)
    _finish(fig, path)


def plot_pairwise_forest(pairwise, split, path):
    sel = pairwise[(pairwise["split"] == split) & (pairwise["case_type"] == "all")
                   & pairwise["metric"].isin(PANEL_METRICS)]
    metrics = [m for m in PANEL_METRICS if m in set(sel["metric"])]
    fig, axes = plt.subplots(1, len(PAIRS), figsize=(4.4 * len(PAIRS), 0.45 * len(metrics) + 1.5), sharey=True)
    for ax, (a, b) in zip(axes, PAIRS):
        rows = sel[(sel["model_a"] == a) & (sel["model_b"] == b)].set_index("metric").loc[metrics]
        rel = 100 * rows["mean_difference"] / rows["mean_b"].abs()
        lo = 100 * rows["ci95_low"] / rows["mean_b"].abs()
        hi = 100 * rows["ci95_high"] / rows["mean_b"].abs()
        y = np.arange(len(metrics))[::-1]
        ax.errorbar(rel, y, xerr=[rel - lo, hi - rel], fmt="o", color=COLORS[a], capsize=3)
        ax.axvline(0, color="k", lw=0.8)
        for yi, (_, r) in zip(y, rows.iterrows()):
            ax.annotate(f"{r['seeds_a_lower']}/{r['n_paired_seeds']}", (hi.loc[r.name], yi),
                        xytext=(4, 0), textcoords="offset points", va="center", fontsize=7)
        ax.set_title(f"{SHORT[a]} − {SHORT[b]}", fontsize=10)
        ax.set_xlabel(f"% of {SHORT[b]} mean (95% CI)")
    axes[0].set_yticks(np.arange(len(metrics))[::-1], [PANEL_METRICS[m] for m in metrics], fontsize=8)
    fig.suptitle(f"Paired differences, all cases, {SPLIT_TITLES[split]}; label = seeds where first model is lower",
                 y=1.0)
    _finish(fig, path)


def plot_validation_curves(hist, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for model in VARIANTS:
        h = hist[hist["model"] == model]
        for ax, col in zip(axes, ("train_loss", "val_mse")):
            g = h.groupby("epoch")[col]
            mean, sd = g.mean(), g.std(ddof=1)
            ax.plot(mean.index, mean.values, color=COLORS[model], lw=1.8, label=MODEL_LABELS[model])
            ax.fill_between(mean.index, mean - sd, mean + sd, color=COLORS[model], alpha=0.15)
    for ax, title in zip(axes, ("train loss", "validation MSE")):
        ax.set_xlabel("epoch")
        ax.set_title(f"{title} (mean ± sd over seeds)")
    axes[0].legend(fontsize=8)
    _finish(fig, path)


# ---------------------------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------------------------

def _p(x):
    return "n/a" if pd.isna(x) else (f"{x:.1e}" if x < 1e-3 else f"{x:.3f}")


def write_summary(path, runs, runs_dirs, seeds, summaries, pairwise, val_table, splits, overviews):
    lines = ["# QLSTM vs Q-sLSTM vs Q-sLSTM-log", "",
             "Sweeps: " + ", ".join(f"`{d}`" for d in runs_dirs), "",
             f"Paired seeds with all three models: {len(seeds)}. Differences are a − b over seeds; "
             "negative = first model lower. CI: Student-t over seed-level differences; p: Wilcoxon signed-rank.", ""]
    recurrences = sorted({(cfg["model"], cfg.get("qslstm_recurrence", "legacy_log_stabilized") if cfg["model"] != "qlstm" else "n/a") for _, cfg in runs.values()})
    lines += ["Recurrences: " + "; ".join(f"{m} = `{r}`" for m, r in recurrences), ""]

    best = val_table.groupby("model")[["best_val_mse", "best_epoch"]].agg(["mean", "std"])
    lines += ["## Validation (best epoch)", "", "| model | best val MSE | best epoch |", "|---|---|---|"]
    for m in VARIANTS:
        r = best.loc[m]
        lines.append(f"| {SHORT[m]} | {r[('best_val_mse', 'mean')]:.5f} ± {r[('best_val_mse', 'std')]:.5f} "
                     f"| {r[('best_epoch', 'mean')]:.1f} ± {r[('best_epoch', 'std')]:.1f} |")
    lines.append("")

    for split in splits:
        s = summaries[split].set_index(["model", "case_type"])
        o = overviews[split].set_index(["model", "case_type"])
        lines += [f"## {SPLIT_TITLES[split]}", "", "### Overview by case type (mean ± sd over seeds)", "",
                  "| model | case | " + " | ".join(OVERVIEW_METRICS.values()) + " |",
                  "|---|---|" + "---|" * len(OVERVIEW_METRICS)]
        for m in VARIANTS:
            for case in CASES:
                cells = [f"{o.loc[(m, case), f'{k}_mean']:.1f} ± {o.loc[(m, case), f'{k}_sd']:.1f}" if k == "best_epoch"
                         else f"{o.loc[(m, case), f'{k}_mean']:.5f} ± {o.loc[(m, case), f'{k}_sd']:.5f}"
                         for k in OVERVIEW_METRICS]
                lines.append(f"| {SHORT[m]} | {case} | " + " | ".join(cells) + " |")
        lines += ["", "### Mean ± sd over seeds (all cases)", "",
                  "| metric | " + " | ".join(SHORT[m] for m in VARIANTS) + " |",
                  "|---|" + "---|" * len(VARIANTS)]
        for metric, label in PANEL_METRICS.items():
            if f"{metric}_mean" not in s.columns:
                continue
            cells = [f"{s.loc[(m, 'all'), f'{metric}_mean']:.5f} ± {s.loc[(m, 'all'), f'{metric}_sd']:.5f}"
                     for m in VARIANTS]
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines += ["", "### MSE by case type", "", "| case | " + " | ".join(SHORT[m] for m in VARIANTS) + " |",
                  "|---|" + "---|" * len(VARIANTS)]
        for case in CASES:
            lines.append(f"| {case} | " + " | ".join(f"{s.loc[(m, case), 'mse_mean']:.5f}" for m in VARIANTS) + " |")
        lines += ["", "### Paired differences", "",
                  "| metric | case | " + " | ".join(f"{SHORT[a]} − {SHORT[b]}" for a, b in PAIRS) + " |",
                  "|---|---|" + "---|" * len(PAIRS)]
        p = pairwise[pairwise["split"] == split].set_index(["metric", "case_type", "model_a", "model_b"])
        rows = [(m, "all") for m in PANEL_METRICS] + [("mse", c) for c in nnm.CASE_ORDER]
        for metric, case in rows:
            if (metric, case, *PAIRS[0]) not in p.index:
                continue
            cells = []
            for a, b in PAIRS:
                r = p.loc[(metric, case, a, b)]
                cells.append(f"{r['mean_difference']:+.5f} ({100 * r['relative_difference']:+.1f}%) "
                             f"[{r['ci95_low']:+.5f}, {r['ci95_high']:+.5f}] "
                             f"{r['seeds_a_lower']}/{r['n_paired_seeds']}, p={_p(r['wilcoxon_p'])}")
            lines.append(f"| {PANEL_METRICS.get(metric, metric)} | {case} | " + " | ".join(cells) + " |")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------

def compare(runs_dirs, out_dir, splits=("test",)):
    out_dir = Path(out_dir)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    runs = collect_runs(runs_dirs)
    ar.MODELS = VARIANTS  # the shared lag / timestep / trace plots iterate over ar.MODELS

    hist = load_history(runs)
    val_table = validation_table(hist)
    val_table.to_csv(out_dir / "best_validation.csv", index=False)
    plot_validation_curves(hist, out_dir / "plots" / "train_val_curves.png")

    splits = [s for s in splits
              if any((d / f"per_seed_metrics_{s}.csv").exists() for d, _ in runs.values())]
    summaries, overviews, pairwise_all, seeds = {}, {}, [], None
    for split in splits:
        per_seed = ar.load_table(runs, f"per_seed_metrics_{split}.csv")
        seeds = paired_seeds(per_seed)
        per_seed = per_seed[per_seed["run_seed"].isin(seeds)]
        per_seed.to_csv(out_dir / f"combined_per_seed_metrics_{split}.csv", index=False)

        summaries[split] = nnm.model_summary(per_seed, list(COMPARISON_METRICS), split)
        summaries[split].to_csv(out_dir / f"model_summary_{split}.csv", index=False)
        overviews[split] = overview_table(per_seed, val_table, split)
        overviews[split].to_csv(out_dir / f"overview_{split}.csv", index=False)
        pairwise = pairwise_table(per_seed, split, COMPARISON_METRICS)
        pairwise.to_csv(out_dir / f"pairwise_comparison_{split}.csv", index=False)
        pairwise_all.append(pairwise)

        plots = out_dir / "plots"
        plot_seed_lines(per_seed, split, plots / f"seed_lines_{split}.png")
        plot_pairwise_forest(pairwise, split, plots / f"pairwise_differences_{split}.png")
        for metric in ("mse", "drift_nonevent", "event_mse"):
            plot_metric_by_case(per_seed, split, metric, plots / f"{metric}_by_case_{split}.png")
        ar.plot_event_curve(per_seed, split, "event_abs_err_lag", "absolute error", "Event-aligned absolute error",
                            plots / f"event_error_curves_{split}.png", cases=("all", *nnm.CASE_ORDER[1:]))
        if "alpha_event" in per_seed.columns:
            ar.plot_event_curve(per_seed, split, "event_alpha_lag", "write proportion alpha", "Event-aligned alpha",
                                plots / f"event_alpha_curves_{split}.png", cases=("all", *nnm.CASE_ORDER[1:]))
        preds = ar.load_table(runs, f"predictions_{split}.csv")
        if not preds.empty:
            preds = preds[preds["run_seed"].isin(seeds)]
            curves = nnm.timestep_curves(preds)
            curves.to_csv(out_dir / f"timestep_curves_{split}.csv", index=False)
            ar.plot_timestep_curves(curves, split, plots / f"timestep_error_drift_{split}.png")
            ar.plot_traces(preds, split, plots / f"traces_{split}.png")

    pairwise = pd.concat(pairwise_all, ignore_index=True)
    (out_dir / "sources.json").write_text(json.dumps(
        {f"{seed}/{model}": str(run_dir) for (seed, model), (run_dir, _) in sorted(runs.items())}, indent=2),
        encoding="utf-8")
    write_summary(out_dir / "summary.md", runs, runs_dirs, seeds, summaries, pairwise, val_table, splits, overviews)
    print(f"comparison written to {out_dir}")
    return out_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description="QLSTM vs the Q-sLSTM variants over several sweeps.")
    parser.add_argument("--runs-dirs", nargs="+", required=True,
                        help="sweep directories; together they must contain every compared model")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS),
                        help="models to compare (default: all; list a subset for sweeps without some models)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--splits", nargs="+", choices=["test", "extrapolation"], default=["test"],
                        help="evaluation splits to compare (default: test only)")
    args = parser.parse_args(argv)
    use_variants(args.variants)
    compare(args.runs_dirs, args.out_dir, args.splits)
    return 0


if __name__ == "__main__":
    sys.exit(main())
