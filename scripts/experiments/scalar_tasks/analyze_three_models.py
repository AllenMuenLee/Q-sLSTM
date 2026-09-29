# scripts/experiments/scalar_tasks/analyze_three_models.py
#
# QLSTM vs Q-sLSTM vs Q-sLSTM (fk) on one scalar-task sweep, mean ± sd over seeds:
#   test_mse.png, test_mae.png, test_rmse.png   overall test metric per task (best-validation checkpoint)
#   mse_vs_timestep.png, mae_vs_timestep.png, rmse_vs_timestep.png
#                                               test error at each supervised timestep
#   loss_vs_epoch.png                           train loss (solid) and validation MSE (dashed) per epoch
#   train_loss_vs_epoch.png, val_loss_vs_epoch.png   the same curves on their own
# Figures use a log y-axis; linear-axis copies are written to linear/.
# plus the CSV tables behind every figure and summary.md.
#
# Per-seed RMSE is sqrt(MSE over all supervised test steps); per-timestep metrics are computed from
# predictions_test.csv over test sequences at that timestep, then summarised across seeds. Only
# seeds on which all models completed every task are used, so comparisons are paired.

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

ROOT = Path(__file__).resolve().parents[3]
TASKS = ("delay", "ema", "running_max", "flip_flop", "narma", "sine_next")
MODELS = ("qlstm", "qslstm", "fk_qslstm")
LABELS = {"qlstm": "QLSTM", "qslstm": "Q-sLSTM", "fk_qslstm": "Q-sLSTM (fk)"}
# Same slots as plot_curves_and_gates.py so the figures read as one set.
COLORS = {"qlstm": "#2a78d6", "qslstm": "#eb6834", "fk_qslstm": "#eda100"}
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
METRIC_LABELS = {"mse": "MSE", "mae": "MAE", "rmse": "RMSE"}
LOG_NOTE = "Log axis: where mean − sd ≤ 0 the band is cut at mean/10."


def discover(runs_dir):
    runs = []
    for complete in sorted(Path(runs_dir).glob("*/seed_*/*/complete.json")):
        run_dir = complete.parent
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        if config.get("model") in MODELS and config.get("task") in TASKS:
            runs.append((config["task"], config["seeds"]["run_seed"], config["model"], run_dir))
    return runs


def paired_seeds(runs):
    have = {}
    for task, seed, model, _ in runs:
        have.setdefault(seed, set()).add((task, model))
    need = {(t, m) for t in TASKS for m in MODELS}
    return sorted(s for s, got in have.items() if need <= got)


def load(runs):
    metrics, timestep, history = [], [], []
    for task, seed, model, run_dir in runs:
        m = pd.read_csv(run_dir / "per_seed_metrics_test.csv")
        metrics.append(m.assign(seed=seed, rmse=np.sqrt(m["mse"]), final_rmse=np.sqrt(m["final_mse"])))

        p = pd.read_csv(run_dir / "predictions_test.csv")
        p = p[p.is_supervised.astype(str) == "True"]
        err = p.prediction - p.target
        g = pd.DataFrame({"timestep": p.timestep, "se": err ** 2, "ae": err.abs()}).groupby("timestep")
        ts = pd.DataFrame({"mse": g.se.mean(), "mae": g.ae.mean()}).reset_index()
        ts["rmse"] = np.sqrt(ts.mse)
        timestep.append(ts.assign(task=task, seed=seed, model=model))

        h = pd.read_csv(run_dir / "history.csv")[["epoch", "train_loss", "val_mse"]]
        history.append(h.assign(task=task, seed=seed, model=model))
    return (pd.concat(metrics, ignore_index=True), pd.concat(timestep, ignore_index=True),
            pd.concat(history, ignore_index=True))


def mean_sd(frame, keys, cols):
    g = frame.groupby(keys)[cols]
    out = g.mean().add_suffix("_mean").join(g.std(ddof=1).add_suffix("_sd")).join(g.size().rename("n_seeds"))
    return out.reset_index()


# ---------------------------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------------------------

def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=MUTED, alpha=0.15, linewidth=0.6)
    ax.set_axisbelow(True)


def task_grid(title):
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.7), facecolor=SURFACE)
    fig.suptitle(title, color=INK, fontsize=13, x=0.01, ha="left")
    for ax in axes.flat:
        style_axes(ax)
    return fig, dict(zip(TASKS, axes.flat))


def finish(fig, out_path, n_seeds, extra_handles=(), note=None, checkpoint=True):
    handles = [plt.Line2D([], [], color=COLORS[m], linewidth=2.5, label=LABELS[m]) for m in MODELS]
    handles += list(extra_handles)
    fig.legend(handles=handles, loc="upper right", ncol=len(handles), frameon=False, fontsize=9,
               labelcolor=INK, bbox_to_anchor=(0.99, 0.995))
    base = f"Mean ± 1 sd over {n_seeds} paired random seeds" + (" (best-validation checkpoint)." if checkpoint else ".")
    fig.text(0.01, 0.005, f"{base} {note}" if note else base, color=MUTED, fontsize=8, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", out_path)


def band(ax, frame, x, y, model, linestyle="-"):
    g = frame[frame.model == model].groupby(x)[y]
    mean, sd = g.mean(), g.std(ddof=1)
    lo = mean - sd
    # Errors and losses are non-negative; on a log axis a band reaching zero is drawn down to mean/10.
    lo = lo.where(lo > 0, mean / 10) if ax.get_yscale() == "log" else lo.clip(lower=0)
    ax.fill_between(mean.index, lo, mean + sd, color=COLORS[model], alpha=0.16, linewidth=0)
    ax.plot(mean.index, mean.values, color=COLORS[model], linewidth=2, linestyle=linestyle)


def bar_plot(metrics, metric, out_path, n_seeds, log_y=False):
    fig, panels = task_grid(f"Test {METRIC_LABELS[metric]} by task" + (" (log scale)" if log_y else ""))
    rng = np.random.default_rng(0)
    for task, ax in panels.items():
        sel = metrics[metrics.task == task]
        if log_y:
            ax.set_yscale("log")
        for k, model in enumerate(MODELS):
            values = sel.loc[sel.model == model, metric].to_numpy()
            mean, sd = values.mean(), values.std(ddof=1)
            lower = min(sd, mean * 0.9) if log_y else sd
            ax.bar(k, mean, width=0.62, color=COLORS[model], alpha=0.85, zorder=2)
            ax.errorbar(k, mean, yerr=[[lower], [sd]], color=INK, capsize=5, linewidth=1.3, zorder=4)
            ax.scatter(k + rng.uniform(-0.18, 0.18, len(values)), values, s=9, color=INK, alpha=0.35,
                       linewidth=0, zorder=3)
        ax.set_xticks(range(len(MODELS)), [LABELS[m] for m in MODELS], fontsize=8, color=MUTED)
        ax.set_title(task, color=INK, fontsize=10, loc="left")
        ax.set_ylabel(f"test {METRIC_LABELS[metric]}", color=MUTED, fontsize=8)
    finish(fig, out_path, n_seeds, note="Bars = mean, whiskers = ± 1 sd, dots = individual seeds.")


def timestep_plot(timestep, metric, out_path, n_seeds, log_y=False):
    fig, panels = task_grid(f"Test {METRIC_LABELS[metric]} by timestep" + (" (log scale)" if log_y else ""))
    for task, ax in panels.items():
        sel = timestep[timestep.task == task]
        if log_y:
            ax.set_yscale("log")
        for model in MODELS:
            band(ax, sel, "timestep", metric, model)
        ax.set_title(task, color=INK, fontsize=10, loc="left")
        ax.set_xlabel("timestep", color=MUTED, fontsize=8)
        ax.set_ylabel(f"test {METRIC_LABELS[metric]}", color=MUTED, fontsize=8)
    note = "Supervised timesteps only (warm-up steps excluded)."
    finish(fig, out_path, n_seeds, note=f"{note} {LOG_NOTE}" if log_y else note)


def epoch_plot(history, columns, title, out_path, n_seeds, note=None, log_y=False):
    fig, panels = task_grid(title + (" (log scale)" if log_y else ""))
    styles = {"train_loss": "-", "val_mse": "--"}
    for task, ax in panels.items():
        sel = history[history.task == task]
        if log_y:
            ax.set_yscale("log")
        for model in MODELS:
            for col in columns:
                band(ax, sel, "epoch", col, model, styles[col])
        ax.set_title(task, color=INK, fontsize=10, loc="left")
        ax.set_xlabel("epoch", color=MUTED, fontsize=8)
        ax.set_ylabel("MSE", color=MUTED, fontsize=8)
    extra = []
    if len(columns) > 1:
        extra = [plt.Line2D([], [], color=MUTED, linewidth=2, linestyle="-", label="train"),
                 plt.Line2D([], [], color=MUTED, linewidth=2, linestyle="--", label="validation")]
    if log_y:
        note = f"{note} {LOG_NOTE}" if note else LOG_NOTE
    finish(fig, out_path, n_seeds, extra, note, checkpoint=False)


# ---------------------------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------------------------

def fmt(mean, sd):
    return f"{mean:.4f} ± {sd:.4f}"


def pairwise(metrics):
    rows = []
    pairs = (("qslstm", "qlstm"), ("fk_qslstm", "qlstm"), ("qslstm", "fk_qslstm"))
    for task in TASKS:
        wide = metrics[metrics.task == task].pivot(index="seed", columns="model")
        for metric in ("mse", "mae", "rmse"):
            for a, b in pairs:
                d = wide[(metric, a)] - wide[(metric, b)]
                p = stats.wilcoxon(d).pvalue if (d != 0).any() else 1.0
                rows.append({"task": task, "metric": metric, "a": a, "b": b, "mean_diff": d.mean(),
                             "sd_diff": d.std(ddof=1), "a_wins": int((d < 0).sum()), "n": len(d),
                             "wilcoxon_p": p})
    return pd.DataFrame(rows)


def write_summary(out, summary, best, pairs, runs_dir, n_seeds):
    lines = ["# QLSTM vs Q-sLSTM vs Q-sLSTM (fk) — scalar tasks", "",
             f"Sweep: `{runs_dir.relative_to(ROOT).as_posix()}`", "",
             f"Paired seeds with all three models on all six tasks: {n_seeds}. Values are mean ± sd over "
             "seeds on the test split, best-validation checkpoint. RMSE per seed = sqrt(test MSE).", "",
             "The per-epoch curves use validation MSE: the runs log train loss and validation MSE each "
             "epoch but evaluate the test split only once, at the best checkpoint.", ""]
    for metric in ("mse", "mae", "rmse"):
        lines += [f"## Test {METRIC_LABELS[metric]}", "", "| task | " + " | ".join(LABELS[m] for m in MODELS)
                  + " | best |", "|---|" + "---|" * (len(MODELS) + 1)]
        for task in TASKS:
            sel = summary[summary.task == task].set_index("model")
            cells = [fmt(sel.loc[m, f"{metric}_mean"], sel.loc[m, f"{metric}_sd"]) for m in MODELS]
            winner = sel.loc[list(MODELS), f"{metric}_mean"].idxmin()
            lines.append(f"| {task} | " + " | ".join(cells) + f" | {LABELS[winner]} |")
        lines.append("")
    lines += ["## Best validation MSE and epoch", "", "| task | " + " | ".join(LABELS[m] for m in MODELS) + " |",
              "|---|" + "---|" * len(MODELS)]
    for task in TASKS:
        sel = best[best.task == task].set_index("model")
        cells = [f"{fmt(sel.loc[m, 'best_val_mse_mean'], sel.loc[m, 'best_val_mse_sd'])} "
                 f"(ep {sel.loc[m, 'best_epoch_mean']:.0f} ± {sel.loc[m, 'best_epoch_sd']:.0f})" for m in MODELS]
        lines.append(f"| {task} | " + " | ".join(cells) + " |")
    lines += ["", "## Paired test-MSE differences (a − b; negative = a lower)", "",
              "Wilcoxon signed-rank over seeds; `wins` = seeds where a has lower MSE.", "",
              "| task | a − b | mean diff ± sd | wins | p |", "|---|---|---|---|---|"]
    for _, r in pairs[pairs.metric == "mse"].iterrows():
        lines.append(f"| {r.task} | {LABELS[r.a]} − {LABELS[r.b]} | {r.mean_diff:+.4f} ± {r.sd_diff:.4f} | "
                     f"{r.a_wins}/{r.n} | {r.wilcoxon_p:.3g} |")
    lines += ["", "## Figures", "", "Log y-axis; linear-axis copies of each are in `linear/`.", ""]
    lines += [f"- `{name}`" for name in ("test_mse.png", "test_mae.png", "test_rmse.png", "mse_vs_timestep.png",
                                         "mae_vs_timestep.png", "rmse_vs_timestep.png", "loss_vs_epoch.png",
                                         "train_loss_vs_epoch.png", "val_loss_vs_epoch.png")]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", out / "summary.md")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "results/scalar_tasks/paper/2026-09-28")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="default: <runs-dir>/analysis/QLSTM vs QsLSTM vs FK_QsLSTM")
    args = parser.parse_args(argv)
    runs_dir = args.runs_dir.resolve()
    out = args.out_dir or runs_dir / "analysis" / "QLSTM vs QsLSTM vs FK_QsLSTM"
    out.mkdir(parents=True, exist_ok=True)

    runs = discover(runs_dir)
    seeds = paired_seeds(runs)
    if not seeds:
        raise SystemExit(f"no seed has all of {MODELS} on all tasks under {runs_dir}")
    runs = [r for r in runs if r[1] in set(seeds)]
    n = len(seeds)
    print(f"{len(runs)} runs, {n} paired seeds")

    metrics, timestep, history = load(runs)
    best = pd.concat([pd.DataFrame([{**json.loads((d / "complete.json").read_text(encoding="utf-8")),
                                     "task": t, "seed": s, "model": m}]) for t, s, m, d in runs],
                     ignore_index=True)

    metrics.to_csv(out / "per_seed_metrics_test.csv", index=False)
    summary = mean_sd(metrics, ["task", "model"], ["mse", "mae", "rmse", "final_mse", "nmse", "r2"])
    summary.to_csv(out / "summary_test.csv", index=False)
    best_summary = mean_sd(best, ["task", "model"], ["best_val_mse", "best_epoch"])
    best_summary.to_csv(out / "best_validation.csv", index=False)
    mean_sd(timestep, ["task", "model", "timestep"], ["mse", "mae", "rmse"]).to_csv(
        out / "timestep_curves_test.csv", index=False)
    mean_sd(history, ["task", "model", "epoch"], ["train_loss", "val_mse"]).to_csv(
        out / "loss_curves.csv", index=False)
    pairs = pairwise(metrics)
    pairs.to_csv(out / "pairwise_comparison_test.csv", index=False)
    (out / "sources.json").write_text(json.dumps(
        {f"{t}/{s}/{m}": str(d.relative_to(ROOT)) for t, s, m, d in runs}, indent=2), encoding="utf-8")

    # fk_qslstm sits orders of magnitude below the others, so the log-axis figures are the main ones;
    # linear-axis copies go to linear/.
    (out / "linear").mkdir(exist_ok=True)
    val_note = "Test split is evaluated only at the best checkpoint, so validation MSE stands in for held-out loss."
    for log_y, folder in ((True, out), (False, out / "linear")):
        for metric in ("mse", "mae", "rmse"):
            bar_plot(metrics, metric, folder / f"test_{metric}.png", n, log_y)
            timestep_plot(timestep, metric, folder / f"{metric}_vs_timestep.png", n, log_y)
        epoch_plot(history, ["train_loss", "val_mse"], "Train and validation loss (masked MSE) by epoch",
                   folder / "loss_vs_epoch.png", n, val_note, log_y)
        epoch_plot(history, ["train_loss"], "Train loss (masked MSE) by epoch",
                   folder / "train_loss_vs_epoch.png", n, log_y=log_y)
        epoch_plot(history, ["val_mse"], "Validation loss (MSE) by epoch",
                   folder / "val_loss_vs_epoch.png", n, val_note, log_y)

    write_summary(out, summary, best_summary, pairs, runs_dir, n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
