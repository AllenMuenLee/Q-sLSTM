# scripts/experiments/nearest_neighbor/plot_model_comparison.py
#
# Collect every seed of one sweep (e.g. results/nearest_neighbor/paper) and draw the QLSTM vs Q-sLSTM
# comparison figures: loss through epochs, MAE / RMSE per split, and the generalization gap.
#
# Per-epoch curves come from history.csv. The training loop does not evaluate the test split each epoch,
# so the per-epoch held-out curve is the validation MSE; the final test MSE of the best-validation
# checkpoint is drawn as a marker on the same axis.
#
# Generalization gaps (per seed, best-validation checkpoint):
#   epoch gap         val_loss_mask_mse - train_loss        (same mask as the training loss, every epoch)
#   train -> test     test mse - train_loss at best epoch   (train loss uses loss_mask; test uses metric_mask)
#   val -> test       test mse - best val mse               (both metric_mask)
#   test -> extrap.   extrapolation mse - test mse          (length generalization)
#
# Seed-to-seed variability: the standard deviation over seeds is drawn per epoch and per metric, with a
# bootstrap 95% CI and a Brown-Forsythe test for unequal spread between the two models.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_results as ar  # noqa: E402
from analyze_results import COLORS, MODEL_LABELS, discover_runs, load_table  # noqa: E402

SPLITS = ("test", "extrapolation")
EPOCH_COLUMNS = {"train_loss": "train loss", "val_mse": "validation MSE", "epoch_gap": "val loss − train loss"}
STD_METRICS = {"test_mae": "test MAE", "test_rmse": "test RMSE", "extrapolation_mae": "extrap. MAE",
               "extrapolation_rmse": "extrap. RMSE", "best_val_mse": "best val MSE"}
N_BOOT = 2000
GAPS = {
    "gap_train_test": "test MSE − train loss",
    "gap_val_test": "test MSE − best val MSE",
    "gap_test_extrap": "extrapolation MSE − test MSE",
}


# ---------------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------------

def load_history(runs):
    frames = []
    for (seed, model), (run_dir, _) in sorted(runs.items()):
        h = pd.read_csv(run_dir / "history.csv")
        h["run_seed"], h["model"] = seed, model
        frames.append(h)
    hist = pd.concat(frames, ignore_index=True)
    hist["epoch_gap"] = hist["val_loss_mask_mse"] - hist["train_loss"]
    return hist


def seed_table(runs, hist):
    """One row per (run_seed, model): MAE / RMSE per split and the generalization gaps."""
    per_seed = pd.concat([load_table(runs, f"per_seed_metrics_{s}.csv") for s in SPLITS], ignore_index=True)
    per_seed = per_seed[per_seed["case_type"] == "all"]
    wide = per_seed.pivot_table(index=["run_seed", "model"], columns="split", values=["mse", "mae"])
    wide.columns = [f"{split}_{metric}" for metric, split in wide.columns]
    wide = wide.reset_index()
    for split in SPLITS:
        if f"{split}_mse" in wide:
            wide[f"{split}_rmse"] = np.sqrt(wide[f"{split}_mse"])

    best = hist.loc[hist.groupby(["run_seed", "model"])["val_mse"].idxmin(),
                    ["run_seed", "model", "epoch", "train_loss", "val_mse"]]
    best = best.rename(columns={"epoch": "best_epoch", "train_loss": "best_train_loss", "val_mse": "best_val_mse"})
    table = wide.merge(best, on=["run_seed", "model"])
    table["gap_train_test"] = table["test_mse"] - table["best_train_loss"]
    table["gap_val_test"] = table["test_mse"] - table["best_val_mse"]
    if "extrapolation_mse" in table:
        table["gap_test_extrap"] = table["extrapolation_mse"] - table["test_mse"]
    return table.sort_values(["model", "run_seed"]).reset_index(drop=True)


def paired_stats(table, metrics):
    """Paired Q-sLSTM − QLSTM differences over seeds that have both models."""
    rows = []
    for metric in metrics:
        if metric not in table:
            continue
        w = table.pivot(index="run_seed", columns="model", values=metric).dropna()
        q = ar.QSLSTM
        d = w[q] - w["qlstm"]
        p = stats.wilcoxon(d).pvalue if len(d) > 1 and (d != 0).any() else np.nan
        rows.append({"metric": metric, "n_pairs": len(d),
                     "qlstm_mean": w["qlstm"].mean(), "qlstm_std": w["qlstm"].std(ddof=1),
                     f"{q}_mean": w[q].mean(), f"{q}_std": w[q].std(ddof=1),
                     "diff_mean": d.mean(), "diff_std": d.std(ddof=1),
                     f"{q}_better_seeds": int((d < 0).sum()), "wilcoxon_p": p,
                     f"std_ratio_{q}_over_qlstm": w[q].std(ddof=1) / w["qlstm"].std(ddof=1),
                     "brown_forsythe_p": stats.levene(w["qlstm"], w[q], center="median").pvalue})
    return pd.DataFrame(rows)


def bootstrap_std_ci(values, n_boot=N_BOOT, seed=0):
    """Percentile 95% CI of the sample standard deviation."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    boots = values[rng.integers(0, len(values), size=(n_boot, len(values)))].std(axis=1, ddof=1)
    return np.percentile(boots, [2.5, 97.5])


# ---------------------------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------------------------

def _finish(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _band(ax, hist, column, model, **kw):
    g = hist[hist["model"] == model].groupby("epoch")[column]
    mean, std = g.mean(), g.std(ddof=1).fillna(0.0)
    ax.plot(mean.index, mean.values, color=COLORS[model], **kw)
    ax.fill_between(mean.index, mean - std, mean + std, color=COLORS[model], alpha=0.22, linewidth=0)


def plot_loss_curves(hist, table, path, log=False):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for model in ar.MODELS:
        n = hist.loc[hist["model"] == model, "run_seed"].nunique()
        _band(axes[0], hist, "train_loss", model, label=f"{MODEL_LABELS[model]} (n={n})")
        _band(axes[1], hist, "val_mse", model, label=f"{MODEL_LABELS[model]} val MSE")
        t = table[table["model"] == model]
        axes[1].errorbar(t["best_epoch"].mean(), t["test_mse"].mean(), yerr=t["test_mse"].std(ddof=1),
                         xerr=t["best_epoch"].std(ddof=1), fmt="*", ms=13, capsize=3, color=COLORS[model],
                         markeredgecolor="black", label=f"{MODEL_LABELS[model]} test MSE @ best epoch")
    axes[0].set_title("Training loss")
    axes[1].set_title("Held-out loss (validation per epoch, test at best checkpoint)")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        if log:
            ax.set_yscale("log")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("MSE (mean ± std over seeds)")
    _finish(fig, path)


def plot_epoch_gap(hist, path, first_epoch=2):
    # Epoch 1's train loss is averaged over a still-untrained model, so it dwarfs the rest of the curve.
    hist = hist[hist["epoch"] >= first_epoch]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for model in ar.MODELS:
        _band(ax, hist, "epoch_gap", model, label=MODEL_LABELS[model])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("epoch")
    ax.set_ylabel("val loss − train loss (mean ± std)")
    ax.set_title(f"Generalization gap through training (epochs ≥ {first_epoch})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    _finish(fig, path)


def _paired_panel(ax, table, metric, title, bars=True):
    """Mean ± std over seeds (bars from zero, or a zoomed marker), grey lines join each seed's two models."""
    w = table.pivot(index="run_seed", columns="model", values=metric).dropna()
    x = np.arange(len(ar.MODELS))
    for _, row in w.iterrows():
        ax.plot(x, row[list(ar.MODELS)].values, color="grey", alpha=0.35, lw=0.8, zorder=1)
    for i, model in enumerate(ar.MODELS):
        if bars:
            ax.bar(i, w[model].mean(), yerr=w[model].std(ddof=1), capsize=4, width=0.55,
                   color=COLORS[model], alpha=0.75, zorder=0)
        else:
            ax.errorbar(i + 0.18, w[model].mean(), yerr=w[model].std(ddof=1), fmt="D", ms=7, capsize=4,
                        color=COLORS[model], markeredgecolor="black", zorder=3)
        ax.scatter(np.full(len(w), i), w[model], color=COLORS[model], edgecolor="black", s=16, zorder=2)
    d = w[ar.QSLSTM] - w["qlstm"]
    p = stats.wilcoxon(d).pvalue if len(d) > 1 and (d != 0).any() else np.nan
    ax.set_xticks(x, [MODEL_LABELS[m].split(" ")[0] for m in ar.MODELS])
    spread = ", ".join(f"{MODEL_LABELS[m].split(' ')[0]} {w[m].mean():.4f}±{w[m].std(ddof=1):.4f}" for m in ar.MODELS)
    ax.set_title(f"{title}\n{spread}\nΔ={d.mean():+.4f}, {MODEL_LABELS[ar.QSLSTM].split(" ")[0]} lower {int((d < 0).sum())}/{len(d)}, p={p:.3g}",
                 fontsize=8.5)
    ax.grid(axis="y", alpha=0.3)


def plot_mae_rmse(table, path):
    splits = [s for s in SPLITS if f"{s}_mae" in table]
    fig, axes = plt.subplots(len(splits), 2, figsize=(9, 4.1 * len(splits)), squeeze=False)
    for r, split in enumerate(splits):
        for c, metric in enumerate(("mae", "rmse")):
            _paired_panel(axes[r, c], table, f"{split}_{metric}", f"{split} {metric.upper()}", bars=False)
    _finish(fig, path)


def plot_gaps(table, path):
    gaps = [g for g in GAPS if g in table]
    fig, axes = plt.subplots(1, len(gaps), figsize=(4.6 * len(gaps), 4.3), squeeze=False)
    for ax, gap in zip(axes[0], gaps):
        _paired_panel(ax, table, gap, GAPS[gap])
        ax.axhline(0, color="black", lw=0.8)
    _finish(fig, path)


def plot_std_epochs(hist, path):
    """Standard deviation over seeds at every epoch (log scale: epoch 1 is far larger than the rest)."""
    fig, axes = plt.subplots(1, len(EPOCH_COLUMNS), figsize=(4.6 * len(EPOCH_COLUMNS), 3.8), squeeze=False)
    for ax, (column, label) in zip(axes[0], EPOCH_COLUMNS.items()):
        for model in ar.MODELS:
            std = hist[hist["model"] == model].groupby("epoch")[column].std(ddof=1)
            ax.plot(std.index, std.values, color=COLORS[model], label=MODEL_LABELS[model])
        ax.set_yscale("log")
        ax.set_title(f"Std over seeds: {label}")
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3, which="both")
    axes[0, 0].set_ylabel("standard deviation")
    axes[0, 0].legend(fontsize=8)
    _finish(fig, path)


def std_table(table, metrics):
    rows = []
    for metric in metrics:
        if metric not in table:
            continue
        for model in ar.MODELS:
            v = table.loc[table["model"] == model, metric].dropna()
            lo, hi = bootstrap_std_ci(v)
            rows.append({"metric": metric, "model": model, "n_seeds": len(v), "std": v.std(ddof=1),
                         "std_ci_low": lo, "std_ci_high": hi})
    return pd.DataFrame(rows)


def plot_std_metrics(stds, summary, path):
    """Across-seed std per metric; error bars are bootstrap 95% CIs, p is Brown-Forsythe."""
    groups = [("metrics", list(STD_METRICS)), ("generalization gaps", list(GAPS))]
    labels = {**STD_METRICS, **{g: t.replace(" − ", " −\n") for g, t in GAPS.items()}}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [len(STD_METRICS), len(GAPS)]})
    width = 0.38
    for ax, (title, metrics) in zip(axes, groups):
        metrics = [m for m in metrics if m in set(stds["metric"])]
        x = np.arange(len(metrics))
        for i, model in enumerate(ar.MODELS):
            sel = stds[stds["model"] == model].set_index("metric").loc[metrics]
            ax.bar(x + (i - 0.5) * width, sel["std"], width, color=COLORS[model], alpha=0.8,
                   label=MODEL_LABELS[model], capsize=3,
                   yerr=[sel["std"] - sel["std_ci_low"], sel["std_ci_high"] - sel["std"]])
        pvals = summary.set_index("metric").loc[metrics, "brown_forsythe_p"]
        ax.set_xticks(x, [f"{labels[m]}\np={pvals[m]:.2g}" for m in metrics], fontsize=8)
        ax.set_title(f"Std over seeds: {title}")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("standard deviation (95% bootstrap CI)")
    axes[0].legend(fontsize=8)
    _finish(fig, path)


# ---------------------------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------------------------

def compare(runs_dir, out_dir=None, qslstm_model="qslstm"):
    ar.use_qslstm_model(qslstm_model)
    runs, incomplete = discover_runs(runs_dir, ar.MODELS)
    if incomplete:
        print(f"WARNING: skipping {len(incomplete)} incomplete run(s): {[str(p) for p in incomplete]}",
              file=sys.stderr)
    seeds = sorted({s for s, _ in runs})
    unpaired = [s for s in seeds if not all((s, m) in runs for m in ar.MODELS)]
    if unpaired:
        print(f"WARNING: seeds {unpaired} lack one model; excluded from paired statistics", file=sys.stderr)
    if not all(any(model == m for _, model in runs) for m in ar.MODELS):
        raise SystemExit(f"need completed runs of both models under {runs_dir}")

    out_dir = Path(out_dir) if out_dir else Path(runs_dir) / ar.analysis_dir_name("analysis") / "model_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    hist = load_history(runs)
    table = seed_table(runs, hist)
    hist.to_csv(out_dir / "combined_history.csv", index=False)
    table.to_csv(out_dir / "seed_table.csv", index=False)
    metrics = [f"{s}_{m}" for s in SPLITS for m in ("mse", "mae", "rmse")] + ["best_val_mse", *GAPS]
    summary = paired_stats(table, metrics)
    summary.to_csv(out_dir / "paired_summary.csv", index=False)
    stds = std_table(table, [*STD_METRICS, *GAPS])
    stds.to_csv(out_dir / "std_summary.csv", index=False)
    epoch_std = hist.groupby(["model", "epoch"])[list(EPOCH_COLUMNS)].agg(["mean", "std"])
    epoch_std.columns = [f"{c}_{a}" for c, a in epoch_std.columns]
    epoch_std.reset_index().to_csv(out_dir / "epoch_mean_std.csv", index=False)

    plot_loss_curves(hist, table, out_dir / "loss_curves.png")
    plot_loss_curves(hist, table, out_dir / "loss_curves_log.png", log=True)
    plot_epoch_gap(hist, out_dir / "generalization_gap_epochs.png")
    plot_mae_rmse(table, out_dir / "mae_rmse.png")
    plot_gaps(table, out_dir / "generalization_gap.png")
    plot_std_epochs(hist, out_dir / "std_through_epochs.png")
    plot_std_metrics(stds, summary, out_dir / "std_comparison.png")

    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(f"{len(seeds)} seeds found ({len(seeds) - len(unpaired)} paired)\n")
        print(summary.to_string(index=False))
        print()
        print(stds.to_string(index=False))
    print(f"\nfigures and tables written to {out_dir}")
    return out_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description="QLSTM vs Q-sLSTM loss / MAE / RMSE / generalization-gap plots.")
    parser.add_argument("--runs-dir", default="results/nearest_neighbor/paper",
                        help="sweep directory containing seed_*/<model>/ runs")
    parser.add_argument("--out-dir", default=None, help="default: <runs-dir>/analysis/model_comparison")
    ar.add_qslstm_model_argument(parser)
    args = parser.parse_args(argv)
    compare(args.runs_dir, args.out_dir, args.qslstm_model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
