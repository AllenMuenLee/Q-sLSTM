# scripts/experiments/scalar_tasks/analyze_results.py
#
# Combine the runs of one sweep into seed tables, per-task model summaries, paired comparisons
# against a reference model, plots, and a machine-generated Markdown summary.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from q_slstm.experiments import scalar_tasks_metrics as stm  # noqa: E402
from q_slstm.datasets.scalar_tasks import TASKS  # noqa: E402
from q_slstm.experiments.scalar_tasks import MODELS  # noqa: E402

MODEL_LABELS = {"qlstm": "QLSTM", "qslstm": "Q-sLSTM", "qslstm_log": "Q-sLSTM-log",
                "qslstm_sqrt": "Q-sLSTM-sqrt", "fk_qslstm": "Q-sLSTM (fk reference)", "fk_qlstm": "QLSTM (fk encoders)", "lstm": "LSTM (classical)"}
COLORS = {"qlstm": "#4C72B0", "qslstm": "#DD8452", "qslstm_log": "#55A868", "fk_qslstm": "#C44E52",
          "fk_qlstm": "#8172B3", "qslstm_sqrt": "#937860",
          "lstm": "#8C8C8C"}
SUMMARY_METRICS = ("mse", "mae", "final_mse", "nmse", "r2", "skill", "baseline_mse")
COMPARISON_METRICS = ("mse", "nmse", "skill", "final_mse")


def discover_runs(runs_dir):
    """Completed runs under `runs_dir` as a list of (run_dir, config); incomplete ones are reported."""
    runs, incomplete = [], []
    for cfg_path in sorted(Path(runs_dir).rglob("config.json")):
        config = json.loads(cfg_path.read_text(encoding="utf-8"))
        if config.get("kind") != "scalar_task_run":
            continue
        if (cfg_path.parent / "complete.json").exists():
            runs.append((cfg_path.parent, config))
        else:
            incomplete.append(cfg_path.parent)
    return runs, incomplete


def load_tables(runs, split):
    per_seed, curves = [], []
    for run_dir, config in runs:
        path = run_dir / f"per_seed_metrics_{split}.csv"
        if not path.exists():
            continue
        per_seed.append(pd.read_csv(path))
        curve = pd.read_csv(run_dir / f"timestep_mse_{split}.csv")
        curves.append(curve.assign(task=config["task"], model=config["model"], run_seed=config["seeds"]["run_seed"]))
    if not per_seed:
        return None, None
    return pd.concat(per_seed, ignore_index=True), pd.concat(curves, ignore_index=True)


def ordered(values, order):
    return [v for v in order if v in set(values)]


# ---------------------------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------------------------

def plot_metric_bars(summary, metric, out_path, title):
    tasks, models = ordered(summary["task"], TASKS), ordered(summary["model"], MODELS)
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(1.6 * len(tasks) + 2, 4))
    x = np.arange(len(tasks))
    for k, model in enumerate(models):
        sel = summary[summary["model"] == model].set_index("task").reindex(tasks)
        ax.bar(x + (k - (len(models) - 1) / 2) * width, sel[f"{metric}_mean"], width,
               yerr=sel[f"{metric}_sd"], capsize=3, color=COLORS.get(model), label=MODEL_LABELS.get(model, model))
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x, tasks)
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_timestep_curves(curves, out_path, title):
    tasks = ordered(curves["task"], TASKS)
    fig, axes = plt.subplots(1, len(tasks), figsize=(3.2 * len(tasks), 3), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        sel = curves[curves["task"] == task]
        for model in ordered(sel["model"], MODELS):
            mean = sel[sel["model"] == model].groupby("timestep")["mse"].mean()
            ax.plot(mean.index, mean.values, color=COLORS.get(model), label=MODEL_LABELS.get(model, model))
        ax.set_title(task)
        ax.set_xlabel("timestep")
        ax.set_yscale("log")
    axes[0][0].set_ylabel("MSE (mean over seeds)")
    axes[0][-1].legend(frameon=False, fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_example_traces(runs, split, out_path, sequence_index=0):
    """Target vs. each model's prediction on one held-out sequence per task (first seed found)."""
    by_task = {}
    for run_dir, config in runs:
        by_task.setdefault(config["task"], {}).setdefault(config["seeds"]["run_seed"], {})[config["model"]] = run_dir
    tasks = ordered(by_task, TASKS)
    if not tasks:
        return
    fig, axes = plt.subplots(len(tasks), 1, figsize=(8, 2.2 * len(tasks)), squeeze=False)
    for ax, task in zip(axes[:, 0], tasks):
        seed = sorted(by_task[task])[0]
        drawn_target = False
        for model in ordered(by_task[task][seed], MODELS):
            path = by_task[task][seed][model] / f"predictions_{split}.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            seq = df[df["sequence_id"] == df["sequence_id"].unique()[sequence_index]]
            if not drawn_target:
                ax.plot(seq["timestep"], seq["input"], color="#BBBBBB", linewidth=0.8, label="input")
                ax.plot(seq["timestep"], seq["target"], color="black", linewidth=1.6, label="target")
                drawn_target = True
            ax.plot(seq["timestep"], seq["prediction"], color=COLORS.get(model), linewidth=1.0,
                    label=MODEL_LABELS.get(model, model))
        ax.set_title(f"{task} (seed {seed})", fontsize=9)
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------

def markdown_table(frame, float_format="{:.4g}"):
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(float_format.format(v) if isinstance(v, float) else str(v)
                                       for v in row) + " |")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Analyze a scalar-task sweep.")
    parser.add_argument("--runs-dir", required=True, help="sweep folder, e.g. results/scalar_tasks/paper/<date>")
    parser.add_argument("--out-dir", default=None, help="default: <runs-dir>/analysis")
    parser.add_argument("--reference-model", choices=MODELS, default="qlstm",
                        help="paired differences are reported as <model> - <reference> (default: qlstm)")
    args = parser.parse_args(argv)

    runs, incomplete = discover_runs(args.runs_dir)
    if not runs:
        raise SystemExit(f"no completed scalar-task runs under {args.runs_dir}")
    out = Path(args.out_dir) if args.out_dir else Path(args.runs_dir) / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    lines = [f"# Scalar-task sweep summary", "", f"Runs: {len(runs)} complete, {len(incomplete)} incomplete.", ""]
    for split in ("test", "extrapolation"):
        per_seed, curves = load_tables(runs, split)
        if per_seed is None:
            continue
        per_seed.to_csv(out / f"per_seed_{split}.csv", index=False)
        summary = stm.model_summary(per_seed, SUMMARY_METRICS, split=split)
        summary.to_csv(out / f"model_summary_{split}.csv", index=False)

        comparisons = [
            stm.paired_difference(per_seed, metric, task, model, args.reference_model, split=split)
            for task in ordered(per_seed["task"], TASKS)
            for model in ordered(per_seed["model"], MODELS) if model != args.reference_model
            for metric in COMPARISON_METRICS
        ]
        comparisons = pd.DataFrame(comparisons)
        if len(comparisons):
            comparisons = comparisons[comparisons["n_paired_seeds"] > 0]
            comparisons.to_csv(out / f"paired_vs_{args.reference_model}_{split}.csv", index=False)

        plot_metric_bars(summary, "skill", out / f"skill_{split}.png",
                         f"Skill vs. constant predictor ({split}); 1 = perfect, 0 = constant")
        plot_metric_bars(summary, "nmse", out / f"nmse_{split}.png", f"Normalized MSE ({split}); lower is better")
        plot_timestep_curves(curves, out / f"timestep_mse_{split}.png", f"Per-timestep MSE ({split})")
        plot_example_traces(runs, split, out / f"example_traces_{split}.png")

        lines += [f"## {split}", "",
                  markdown_table(summary[["task", "model", "n_seeds", "mse_mean", "mse_sd",
                                          "nmse_mean", "skill_mean", "skill_sd"]]), ""]
        if len(comparisons):
            lines += [f"### Paired differences vs. {args.reference_model} (negative mse/nmse = better; "
                      f"positive skill = better)", "",
                      markdown_table(comparisons[["task", "model_a", "metric", "n_paired_seeds",
                                                  "mean_difference", "ci95_low", "ci95_high"]]), ""]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"analysis written to {out}")


if __name__ == "__main__":
    main()
