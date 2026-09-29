# scripts/experiments/nearest_neighbor/plot_key_results.py
#
# Four headline figures for QLSTM vs Q-sLSTM vs Q-sLSTM-log on the test split:
#
#   revision_gain_vs_gap   every candidate: |p[t-1] - v[t]| - |p[t] - v[t]|, the move toward candidate t's value v[t]
#                          (on new records v[t] is the new target, so this is revision_gain), against the signed gap
#                          similarity[t] - running best before t (< 0: not a record, > 0: new record)
#   drift_vs_gap           every candidate: |p[t] - b[t]| - |p[t-1] - b[t]|, the move away from the best value b[t] held
#                          before candidate t (on non-events b[t] is the unchanged target; on new records the old best,
#                          which the model should now leave), against the same signed gap
#   error_vs_gap           MSE / MAE / RMSE of p[t] against the true target y[t], against the same signed gap
#   test_metrics           MSE / MAE / RMSE of the best-validation checkpoint
#   train_loss             training loss per epoch
#   generalization_gap     val loss - train loss per epoch, and per seed at the best-validation checkpoint:
#                          test MSE - train loss, test MSE - best val MSE, extrapolation MSE - test MSE
#
# Only steps whose predecessor is a metric step are used, as for revision_gain in nearest_neighbor_metrics.
# Every value is first averaged within a seed; bands and error bars are over seeds.

from __future__ import annotations

import argparse
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

import analyze_results as ar  # noqa: E402
import false_revision as fr  # noqa: E402
import plot_model_comparison as pmc  # noqa: E402
from scipy import stats  # noqa: E402
from analyze_results import COLORS, MODEL_LABELS  # noqa: E402
from compare_variants import PAIRS, SHORT, VARIANTS, collect_runs, load_history, paired_seeds  # noqa: E402

SIGNED_BINS = [-2.0, -1.0, -0.5, -0.2, -0.1, -0.05, -0.02, 0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 2.0]
SPLIT = "test"


def load_steps(runs, seeds):
    cache, frames = {}, []
    for (seed, model), (run_dir, config) in sorted(runs.items()):
        if seed not in seeds:
            continue
        steps = fr.step_table(run_dir, config, SPLIT, cache)
        # best value held before step t, taken from the unfiltered predictions (step_table drops step t-1
        # when it is not itself preceded by a metric step)
        preds = pd.read_csv(run_dir / f"predictions_{SPLIT}.csv", usecols=["sequence_id", "timestep", "target"])
        preds = preds.sort_values(["sequence_id", "timestep"])
        preds["prev_target"] = preds.groupby("sequence_id")["target"].shift()
        frames.append(steps.merge(preds[["sequence_id", "timestep", "prev_target"]],
                                  on=["sequence_id", "timestep"], validate="one_to_one"))
    steps = pd.concat(frames, ignore_index=True)
    steps["signed_gap"] = steps["similarity"] - steps["prev_best"]
    steps["pull"] = (steps["prev_pred"] - steps["value"]).abs() - (steps["prediction"] - steps["value"]).abs()
    steps["drift"] = (steps["prediction"] - steps["prev_target"]).abs() - (steps["prev_pred"] - steps["prev_target"]).abs()
    return steps


def binned(steps, value):
    """Per seed mean of `value` in each signed-gap bin, then mean / sd / count over seeds."""
    sel = steps.assign(bin=pd.cut(steps["signed_gap"], SIGNED_BINS, include_lowest=True))
    per_seed = (sel.groupby(["model", "run_seed", "bin"], observed=True)
                .agg(value=(value, "mean"), n_steps=(value, "size"), gap=("signed_gap", "mean")).reset_index())
    out = (per_seed.groupby(["model", "bin"], observed=True)
           .agg(mean=("value", "mean"), sd=("value", "std"), gap=("gap", "mean"),
                n_seeds=("value", "size"), steps_per_seed=("n_steps", "mean")).reset_index())
    out["bin"] = out["bin"].astype(str)
    return out


def plot_vs_signed_gap(table, ylabel, title, left_note, right_note, path):
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.axvspan(0, 2, color="0.92", zorder=0)
    ax.text(0.03, 0.97, right_note, transform=ax.get_xaxis_transform(), va="top", fontsize=8, color="0.3")
    ax.text(-0.03, 0.97, left_note, transform=ax.get_xaxis_transform(), va="top", ha="right", fontsize=8, color="0.3")
    for model in VARIANTS:
        t = table[table["model"] == model]
        ax.plot(t["gap"], t["mean"], marker="o", lw=2, color=COLORS[model], label=MODEL_LABELS[model])
        ax.fill_between(t["gap"], t["mean"] - t["sd"], t["mean"] + t["sd"], color=COLORS[model], alpha=0.15)
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(0, color="k", lw=0.8, ls=":")
    ax.set_xscale("symlog", linthresh=0.02)
    ax.set_xlim(-2, 2)
    ax.set_xlabel("current similarity − running best  (symlog; left = far below the best, right = new record)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="center left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def error_by_gap(steps):
    """Per seed MSE / MAE / RMSE of p[t] vs the true target in each signed-gap bin, then mean / sd over seeds."""
    sel = steps.assign(bin=pd.cut(steps["signed_gap"], SIGNED_BINS, include_lowest=True),
                       sq=(steps["prediction"] - steps["target"]) ** 2,
                       ab=(steps["prediction"] - steps["target"]).abs())
    per_seed = (sel.groupby(["model", "run_seed", "bin"], observed=True)
                .agg(mse=("sq", "mean"), mae=("ab", "mean"), gap=("signed_gap", "mean"), n_steps=("sq", "size"))
                .reset_index())
    per_seed["rmse"] = np.sqrt(per_seed["mse"])
    out = per_seed.groupby(["model", "bin"], observed=True).agg(
        gap=("gap", "mean"), steps_per_seed=("n_steps", "mean"),
        **{f"{m}_{s}": (m, s) for m in ("mse", "mae", "rmse") for s in ("mean", "std")}).reset_index()
    out["bin"] = out["bin"].astype(str)
    return out


def plot_error_vs_signed_gap(table, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharex=True)
    for ax, (metric, label) in zip(axes, (("mse", "MSE"), ("mae", "MAE"), ("rmse", "RMSE"))):
        ax.axvspan(0, 2, color="0.92", zorder=0)
        for model in VARIANTS:
            t = table[table["model"] == model]
            mean, sd = t[f"{metric}_mean"], t[f"{metric}_std"]
            ax.plot(t["gap"], mean, marker="o", lw=2, color=COLORS[model], label=MODEL_LABELS[model])
            ax.fill_between(t["gap"], mean - sd, mean + sd, color=COLORS[model], alpha=0.15)
        ax.axvline(0, color="k", lw=0.8, ls=":")
        ax.set_xscale("symlog", linthresh=0.02)
        ax.set_xlim(-2, 2)
        ax.set_xlabel("current similarity − running best  (symlog)")
        ax.set_title(f"{label} at the step")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Prediction error vs similarity to the current best (left = not a record, shaded = new record; "
                 "band = sd over seeds)", y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def test_metric_table(runs, seeds):
    per_seed = ar.load_table(runs, f"per_seed_metrics_{SPLIT}.csv")
    per_seed = per_seed[(per_seed["case_type"] == "all") & per_seed["run_seed"].isin(seeds)].copy()
    per_seed["rmse"] = np.sqrt(per_seed["mse"])
    return per_seed[["run_seed", "model", "mse", "mae", "rmse"]]


def plot_test_metrics(per_seed, path):
    metrics = {"mse": "MSE", "mae": "MAE", "rmse": "RMSE"}
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    x = np.arange(len(VARIANTS))
    for ax, (metric, label) in zip(axes, metrics.items()):
        for k, model in enumerate(VARIANTS):
            v = per_seed[per_seed["model"] == model][metric]
            ax.bar(k, v.mean(), 0.6, yerr=v.std(ddof=1), capsize=5, color=COLORS[model], alpha=0.6)
            ax.scatter(np.full(len(v), k), v, color=COLORS[model], edgecolor="k", s=14, zorder=3)
            ax.annotate(f"{v.mean():.4f}\n± {v.std(ddof=1):.4f}", (k, v.max()), xytext=(0, 6),
                        textcoords="offset points", ha="center", fontsize=8)
        lo = per_seed[metric].min()
        hi = per_seed[metric].max()
        ax.set_ylim(lo - 0.25 * (hi - lo), hi + 0.45 * (hi - lo))
        ax.set_xticks(x, [SHORT[m] for m in VARIANTS])
        ax.set_title(f"test {label}")
    fig.suptitle("Test error of the best-validation checkpoint (bar = mean ± sd over seeds, dots = seeds)", y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_train_loss(hist, path):
    fig, ax = plt.subplots(figsize=(7, 4.3))
    for model in VARIANTS:
        g = hist[(hist["model"] == model) & (hist["epoch"] >= 2)].groupby("epoch")["train_loss"]
        mean, sd = g.mean(), g.std(ddof=1)
        ax.plot(mean.index, mean.values, color=COLORS[model], lw=2, label=MODEL_LABELS[model])
        ax.fill_between(mean.index, mean - sd, mean + sd, color=COLORS[model], alpha=0.15)
    ax.set_xlabel("epoch")
    ax.set_ylabel("train loss (masked MSE)")
    ax.set_title("Training loss (mean ± sd over seeds; epoch 1 omitted, ≈0.14)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_generalization_gap(hist, table, path, first_epoch=2):
    """Epoch gap bands plus one panel per checkpoint gap; grey lines join each seed's three runs."""
    gaps = [g for g in pmc.GAPS if g in table]
    fig, axes = plt.subplots(1, 1 + len(gaps), figsize=(4.4 * (1 + len(gaps)), 4.6),
                             gridspec_kw={"width_ratios": [1.4] + [1] * len(gaps)})
    ax = axes[0]
    for model in VARIANTS:
        g = hist[(hist["model"] == model) & (hist["epoch"] >= first_epoch)].groupby("epoch")["epoch_gap"]
        mean, sd = g.mean(), g.std(ddof=1)
        ax.plot(mean.index, mean.values, color=COLORS[model], lw=2, label=MODEL_LABELS[model])
        ax.fill_between(mean.index, mean - sd, mean + sd, color=COLORS[model], alpha=0.15)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("epoch")
    ax.set_ylabel("gap (mean ± sd over seeds)")
    ax.set_title(f"val loss − train loss through training (epochs ≥ {first_epoch})", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    x = np.arange(len(VARIANTS))
    for ax, gap in zip(axes[1:], gaps):
        w = table.pivot(index="run_seed", columns="model", values=gap)[list(VARIANTS)].dropna()
        for _, row in w.iterrows():
            ax.plot(x, row.values, color="grey", alpha=0.35, lw=0.8, zorder=1)
        for k, model in enumerate(VARIANTS):
            ax.bar(k, w[model].mean(), 0.6, yerr=w[model].std(ddof=1), capsize=5, color=COLORS[model], alpha=0.6)
            ax.scatter(np.full(len(w), k), w[model], color=COLORS[model], edgecolor="k", s=14, zorder=3)
        tests = []
        for a, b in PAIRS:
            d = w[a] - w[b]
            p = stats.wilcoxon(d).pvalue if (d != 0).any() else np.nan
            tests.append(f"{SHORT[a]}−{SHORT[b]}: {d.mean():+.4f}, p={p:.2g}")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x, [SHORT[m] for m in VARIANTS], fontsize=8)
        ax.set_title(f"{pmc.GAPS[gap]} (n={len(w)})\n" + "\n".join(tests), fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Generalization gap (best-validation checkpoint; bar = mean ± sd over seeds, Wilcoxon signed-rank)",
                 y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Headline figures for QLSTM / Q-sLSTM / Q-sLSTM-log.")
    parser.add_argument("--runs-dirs", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--min-timestep", type=int, default=None,
                        help="only redraw revision gain / drift from steps t >= this, as *_t<N> files")
    args = parser.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    runs = collect_runs(args.runs_dirs)
    ar.MODELS = VARIANTS
    metrics = test_metric_table(runs, paired_seeds(ar.load_table(runs, f"per_seed_metrics_{SPLIT}.csv")))
    seeds = set(metrics["run_seed"])

    steps = load_steps(runs, seeds)
    # Early steps write large fractions (small normalizer) and hold most large records; --min-timestep drops them.
    suffix, note = "", ""
    if args.min_timestep is not None:
        steps = steps[steps["timestep"] >= args.min_timestep]
        suffix, note = f"_t{args.min_timestep}", f"\n(steps t ≥ {args.min_timestep} only)"
    gain = binned(steps, "pull")
    gain.to_csv(out / f"revision_gain_vs_gap{suffix}.csv", index=False)
    plot_vs_signed_gap(gain, "move toward the candidate's value",
                       f"Revision gain for every candidate vs its similarity to the current best{note}",
                       "← not a record\nshould ignore (≈ 0)", "new record →\nshould move to its value",
                       out / f"revision_gain_vs_gap{suffix}.png")
    drift = binned(steps, "drift")
    drift.to_csv(out / f"drift_vs_gap{suffix}.csv", index=False)
    plot_vs_signed_gap(drift, "move away from the best held before the step",
                       f"Drift from the current best for every candidate vs its similarity to the current best{note}",
                       "← not a record\nshould stay on the best (≈ 0)", "new record →\nshould leave the old best",
                       out / f"drift_vs_gap{suffix}.png")
    if args.min_timestep is not None:
        for name, table in (("revision gain", gain), ("drift", drift)):
            print(f"== {name} ==")
            print(table.pivot(index="gap", columns="model", values="mean").round(4).to_string())
        print(gain.pivot(index="gap", columns="model", values="steps_per_seed").round(1).to_string())
        print(f"figures written to {out}")
        return 0
    errors = error_by_gap(steps)
    errors.to_csv(out / "error_vs_gap.csv", index=False)
    plot_error_vs_signed_gap(errors, out / "error_vs_gap.png")

    metrics.to_csv(out / "test_metrics_per_seed.csv", index=False)
    summary = metrics.groupby("model")[["mse", "mae", "rmse"]].agg(["mean", "std"]).loc[list(VARIANTS)]
    summary.to_csv(out / "test_metrics_summary.csv")
    plot_test_metrics(metrics, out / "test_metrics.png")

    hist = load_history({k: v for k, v in runs.items() if k[0] in seeds})
    plot_train_loss(hist, out / "train_loss.png")

    gap_hist = pmc.load_history({k: v for k, v in runs.items() if k[0] in seeds})
    gap_table = pmc.seed_table({k: v for k, v in runs.items() if k[0] in seeds}, gap_hist)
    gap_cols = ["run_seed", "model", "best_epoch", *[g for g in pmc.GAPS if g in gap_table]]
    gap_table[gap_cols].to_csv(out / "generalization_gap_per_seed.csv", index=False)
    plot_generalization_gap(gap_hist, gap_table, out / "generalization_gap.png")
    print(summary.round(5).to_string())
    print(f"figures written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
