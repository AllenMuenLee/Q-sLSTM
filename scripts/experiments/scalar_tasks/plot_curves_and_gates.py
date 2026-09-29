# scripts/experiments/scalar_tasks/plot_curves_and_gates.py
#
# Figures for one scalar-task sweep folder (completed runs only):
#   test_r2.png                 test R^2 per seed, per task and model
#   test_mse_vs_timestep.png    test MSE at each timestep (median over seeds, IQR band)
#   train_loss.png              training loss per epoch (median over seeds, IQR band)
#   input_gate_vs_timestep.png  input gate at each test timestep (best checkpoint replayed)
#   forget_gate_vs_timestep.png forget gate at each test timestep
#
# Gate values are the raw gates of each recurrence, averaged over hidden units and test sequences:
#   qlstm       i = sigmoid(q_i)                f = sigmoid(q_f)
#   qslstm      i = (1+q_i)/(1-q_i) (eps-clamped) f = (1+q_f)/(1-q_f)  (sigmoid(q_f) in sigmoid-forget runs)
#   qslstm_log  i = ln(2/(1-q_i))               f = ln(2/(1-q_f))    (sigmoid(q_f) in sigmoid-forget runs)
#   fk_qslstm   i = exp(Linear(VQC(Linear(.))))  f = exp(Linear(VQC(Linear(.))))  (unbounded)
#   fk_qlstm    i = sigmoid(Linear(VQC(Linear(.))))  f = sigmoid(Linear(VQC(Linear(.))))
# Input gates live on different scales, so that figure uses a log axis. Gate traces are cached in
# <runs-dir>/figures/gate_traces.csv; pass --retrace to recompute them.

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(ROOT / "src"))

from q_slstm.datasets.scalar_tasks import TASKS as SCALAR_TASKS  # noqa: E402

# Panel order; solar_next sweeps (scripts/experiments/solar_next) reuse these figures.
TASKS = (*SCALAR_TASKS, "solar_next")

MODELS = ("qlstm", "qslstm", "qslstm_log", "fk_qslstm", "fk_qlstm")
LABELS = {"qlstm": "QLSTM", "qslstm": "Q-sLSTM", "qslstm_log": "Q-sLSTM-log", "fk_qslstm": "Q-sLSTM (fk)",
          "fk_qlstm": "QLSTM (fk)"}
# Validated categorical slots 1-5 (blue, orange, aqua, yellow, magenta) on a light surface.
COLORS = {"qlstm": "#2a78d6", "qslstm": "#eb6834", "qslstm_log": "#1baf7a", "fk_qslstm": "#eda100",
          "fk_qlstm": "#e87ba4"}
FK_MODELS = ("fk_qslstm", "fk_qlstm")
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"


def discover(runs_dir):
    """(task, seed, model, run_dir) of completed runs; task and seed come from each run's config.json,
    so both <task>/seed_*/<model> (scalar tasks) and seed_*/<model> (solar_next) layouts work."""
    runs = []
    for complete in sorted(Path(runs_dir).rglob("complete.json")):
        run_dir = complete.parent
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        if config.get("model") in MODELS and config.get("task") in TASKS:
            runs.append((config["task"], config["seeds"]["run_seed"], config["model"], run_dir))
    return runs


def load_test_data(config):
    """The run's test split, built by the experiment that produced it."""
    if config.get("kind") == "solar_next_run":
        from q_slstm.experiments.solar_next import make_datasets
    else:
        from q_slstm.experiments.scalar_tasks import make_datasets
    return make_datasets({**config, "run_extrapolation": False})["test"]


# ---------------------------------------------------------------------------------------------
# Gate replay
# ---------------------------------------------------------------------------------------------

def trace_gates(job):
    """Mean input/forget gate per timestep over `n_sequences` test sequences, best checkpoint."""
    import torch

    from q_slstm.experiments import scalar_tasks as st
    from q_slstm.models.factory import match_recurrence
    from q_slstm.models.q_slstm_cell import QSLSTM_SIGMOID_FORGET_RECURRENCE, bounded_log_ratio
    from q_slstm.models.q_slstm_log_cell import (
        QSLSTM_LOG_SIGMOID_FORGET_RECURRENCES, float64_expectations, logarithmic_gate,
    )

    task, seed, model_name, run_dir, n_sequences = job
    torch.set_num_threads(1)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    test = load_test_data(config)
    model = st.build_model(config)
    state = torch.load(run_dir / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    match_recurrence(model, config.get("qslstm_recurrence"))
    cell = model.cell
    sigmoid_forget = config.get("qslstm_recurrence") in (QSLSTM_SIGMOID_FORGET_RECURRENCE,
                                                         *QSLSTM_LOG_SIGMOID_FORGET_RECURRENCES)

    x = test.tensors["inputs"][:n_sequences]
    batch, length, _ = x.shape
    n_states = 2 if model_name in ("qlstm", *FK_MODELS) else 4
    hidden = tuple(x.new_zeros(batch, cell.hidden_size) for _ in range(n_states))
    rows = []
    with torch.no_grad():
        for t in range(length):
            combined = torch.cat((x[:, t], hidden[0]), dim=-1)
            if model_name in FK_MODELS:
                i_gate, f_gate, _, _ = cell.gate_values(combined)
            elif model_name == "qslstm_log":
                q_i, q_f = (float64_expectations(g, combined) for g in (cell.input_gate, cell.forget_gate))
                i_gate = logarithmic_gate(q_i)
            else:
                q_i, q_f = cell.input_gate(combined), cell.forget_gate(combined)
                i_gate = (torch.sigmoid(q_i) if model_name == "qlstm"
                          else torch.exp(bounded_log_ratio(q_i, cell.gate_epsilon)))
            if model_name == "qlstm" or (model_name not in FK_MODELS and sigmoid_forget):
                f_gate = torch.sigmoid(q_f)
            elif model_name == "qslstm":
                f_gate = torch.exp(bounded_log_ratio(q_f, cell.gate_epsilon))
            elif model_name == "qslstm_log":
                f_gate = logarithmic_gate(q_f)
            rows.append({"task": task, "seed": seed, "model": model_name, "timestep": t,
                         "input_gate": float(i_gate.double().mean()), "forget_gate": float(f_gate.double().mean())})
            _, *hidden = cell(x[:, t], tuple(hidden))
    return rows


def gate_traces(runs, cache, n_sequences, workers, retrace):
    if cache.exists() and not retrace:
        cached = pd.read_csv(cache)
        done = set(zip(cached.task, cached.seed, cached.model))
    else:
        cached, done = pd.DataFrame(), set()
    jobs = [(t, s, m, d, n_sequences) for t, s, m, d in runs if (t, s, m) not in done]
    if jobs:
        print(f"replaying {len(jobs)} checkpoints on {n_sequences} test sequences ({workers} workers)", flush=True)
        rows = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for k, result in enumerate(pool.map(trace_gates, jobs), 1):
                rows += result
                if k % 20 == 0 or k == len(jobs):
                    print(f"  {k}/{len(jobs)}", flush=True)
        cached = pd.concat([cached, pd.DataFrame(rows)], ignore_index=True)
        cached.to_csv(cache, index=False)
    return cached


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


def task_grid(title, frame):
    """One panel per task present in `frame`, three per row."""
    tasks = [t for t in TASKS if t in set(frame.task)]
    cols = min(3, len(tasks))
    rows = -(-len(tasks) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 3.5 * rows + 0.7), facecolor=SURFACE,
                             squeeze=False)
    fig.suptitle(title, color=INK, fontsize=13, x=0.01, ha="left")
    for ax in axes.flat[len(tasks):]:
        ax.set_visible(False)
    for ax in axes.flat:
        style_axes(ax)
    return fig, dict(zip(tasks, axes.flat))


def finish(fig, models, counts, out_path, note=None):
    handles = [plt.Line2D([], [], color=COLORS[m], linewidth=2, marker="o", markersize=5,
                          label=f"{LABELS[m]} (n={counts[m]})") for m in models]
    fig.legend(handles=handles, loc="upper right", ncol=len(models), frameon=False, fontsize=9,
               labelcolor=INK, bbox_to_anchor=(0.99, 0.995))
    if note:
        fig.text(0.01, 0.005, note, color=MUTED, fontsize=8, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", out_path)


def band_plot(frame, x, y, title, xlabel, ylabel, out_path, models, counts, log_y=False, note=None):
    """Median over seeds per x, with the interquartile range as a band."""
    fig, panels = task_grid(title, frame)
    for task, ax in panels.items():
        sel = frame[frame.task == task]
        for model in models:
            g = sel[sel.model == model].groupby(x)[y]
            if not len(g):
                continue
            med, lo, hi = g.median(), g.quantile(0.25), g.quantile(0.75)
            ax.fill_between(med.index, lo, hi, color=COLORS[model], alpha=0.18, linewidth=0)
            ax.plot(med.index, med.values, color=COLORS[model], linewidth=2)
        ax.set_title(task, color=INK, fontsize=10, loc="left")
        ax.set_xlabel(xlabel, color=MUTED, fontsize=8)
        ax.set_ylabel(ylabel, color=MUTED, fontsize=8)
        if log_y:
            ax.set_yscale("log")
    finish(fig, models, counts, out_path, note)


def strip_plot(metrics, out_path, models, counts):
    fig, panels = task_grid("Test R² per seed (best-validation checkpoint); bar = median", metrics)
    rng = np.random.default_rng(0)
    for task, ax in panels.items():
        sel = metrics[metrics.task == task]
        for k, model in enumerate(models):
            values = sel.loc[sel.model == model, "r2"].to_numpy()
            if not len(values):
                continue
            jitter = rng.uniform(-0.15, 0.15, len(values))
            ax.scatter(k + jitter, values, s=22, color=COLORS[model], edgecolor=SURFACE, linewidth=1, zorder=3)
            ax.hlines(np.median(values), k - 0.3, k + 0.3, color=INK, linewidth=2, zorder=4)
        ax.set_xticks(range(len(models)), [LABELS[m] for m in models], fontsize=8, color=MUTED)
        ax.set_title(task, color=INK, fontsize=10, loc="left")
        ax.set_ylabel("test R²", color=MUTED, fontsize=8)
    finish(fig, models, counts, out_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, required=True, help="a sweep folder, e.g. results/scalar_tasks/paper/2026-09-28")
    parser.add_argument("--n-sequences", type=int, default=128, help="test sequences replayed per run for gate traces")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retrace", action="store_true", help="recompute cached gate traces")
    args = parser.parse_args(argv)

    runs = discover(args.runs_dir)
    if not runs:
        raise SystemExit(f"no completed runs under {args.runs_dir}")
    out = args.runs_dir / "figures"
    out.mkdir(exist_ok=True)
    models = [m for m in MODELS if any(r[2] == m for r in runs)]
    counts = {m: sum(r[2] == m for r in runs) for m in models}

    metrics = pd.concat([pd.read_csv(d / "per_seed_metrics_test.csv").assign(task=t, model=m)
                         for t, _, m, d in runs], ignore_index=True)
    timestep = pd.concat([pd.read_csv(d / "timestep_mse_test.csv").assign(task=t, seed=s, model=m)
                          for t, s, m, d in runs], ignore_index=True)
    history = pd.concat([pd.read_csv(d / "history.csv").assign(task=t, seed=s, model=m)
                         for t, s, m, d in runs], ignore_index=True)
    note = "n = completed runs across all tasks; lines = median over seeds, bands = interquartile range."

    strip_plot(metrics, out / "test_r2.png", models, counts)
    band_plot(timestep, "timestep", "mse", "Test MSE by timestep", "timestep", "test MSE",
              out / "test_mse_vs_timestep.png", models, counts, log_y=True, note=note)
    band_plot(history, "epoch", "train_loss", "Training loss (masked MSE)", "epoch", "train loss",
              out / "train_loss.png", models, counts, log_y=True, note=note)

    gates = gate_traces(runs, out / "gate_traces.csv", args.n_sequences, args.workers, args.retrace)
    band_plot(gates, "timestep", "input_gate", "Input gate by timestep (test data, best checkpoint)",
              "timestep", "input gate (log scale)", out / "input_gate_vs_timestep.png", models, counts,
              log_y=True, note="QLSTM: sigmoid(q); Q-sLSTM: (1+q)/(1-q); Q-sLSTM-log: ln(2/(1-q)); "
                               "Q-sLSTM (fk): exp(encoder output); QLSTM (fk): sigmoid(encoder output). "
                               "Mean over hidden units and test sequences. " + note)
    band_plot(gates, "timestep", "forget_gate", "Forget gate by timestep (test data, best checkpoint)",
              "timestep", "forget gate", out / "forget_gate_vs_timestep.png", models, counts,
              note="QLSTM: sigmoid(q); Q-sLSTM: (1+q)/(1-q); Q-sLSTM-log: ln(2/(1-q)) (sigmoid(q) in "
                   "sigmoid-forget runs); fk models: of their encoder output. "
                   "Mean over hidden units and test sequences. " + note)
    return 0


if __name__ == "__main__":
    sys.exit(main())
