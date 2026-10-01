# scripts/experiments/scalar_tasks/analyze_qlstm_vs_qslstm.py
#
# QLSTM vs Q-sLSTM (odd-gate recurrence, (1+q)/(1-q) gates) on one scalar-task sweep, paired over seeds.
# Writes to <runs-dir>/analysis/QLSTM vs QsLSTM (odd gate)/:
#
#   metrics_per_seed.csv      every metric below, one row per task/seed/model
#   comparison.csv            per task and metric: both models' mean/sd/median, Q-sLSTM/QLSTM geometric-
#                             mean ratio with a paired-bootstrap 95% CI, Wilcoxon p (Holm-adjusted over
#                             tasks), Q-sLSTM wins, Cohen's d_z and rank-biserial effect size
#   curves_epoch.csv, curves_timestep.csv   median/IQR curves behind the figures
#   summary.md                the tables in prose order
#   fig_overview.png          effect-size heatmap: every metric x task, oriented so + = Q-sLSTM better
#   fig_test_mse_paired.png   test MSE per seed, paired seeds joined
#   fig_error_cdf.png         distribution of absolute test errors
#   fig_timestep.png          test MSE per timestep at the training length (32) and at 128
#   fig_val_curves.png        validation MSE per epoch with the shared convergence threshold
#   fig_convergence.png       epochs to the shared threshold, epochs to 90% of own improvement, AUC
#   fig_generalization.png    MSE on train / val / test / longer sequences / noisy inputs
#
# Metric groups
#   accuracy      test MSE, RMSE, MAE, median / p95 / max absolute error, final-step MSE, NMSE, R², skill
#                 vs the constant training-mean baseline, Pearson r, bias (mean error), error sd,
#                 worst-sequence and p90-sequence MSE, directional accuracy (sign of the predicted
#                 step change), flip_flop latch accuracy (|error| < 0.5 where a bit is held)
#   convergence   validation MSE at epochs 1/5/10/20/40/60, log-AUC of the validation curve, epochs
#                 (and wall-clock seconds) to reach a threshold shared by both models, epochs to 90%
#                 of the model's own log-improvement, best epoch, late train-loss slope
#   stability     validation spikes (> 2x running minimum), late-epoch log-roughness, gradient norms,
#                 spread over seeds (sd, CV, worst seed, seeds with R² < 0.5)
#   generalization  replayed best checkpoint: train-set MSE (first 400 training sequences), test/train ratio, validation/train
#                 ratio, MSE at length 64 and 128 (4x the training length) and their ratios to length
#                 32, late/early error ratio at length 128, test MSE with Gaussian input noise
#                 (sd 0.05, 0.1) and its ratio to clean
#   cost          trainable parameters, seconds per epoch, total training seconds
#
# The replay (inference only, no training) is cached in replay_metrics.csv; --no-replay skips it,
# --rereplay recomputes it.

from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

TASKS = ("delay", "ema", "running_max", "flip_flop", "narma", "sine_next")
A, B = "qlstm", "qslstm"  # ratios and differences are B relative to A
MODELS = (A, B)
LABELS = {A: "QLSTM", B: "Q-sLSTM"}
COLORS = {A: "#2a78d6", B: "#eb6834"}  # same slots as the other scalar-task figures
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
OUT_NAME = "QLSTM vs QsLSTM (odd gate)"
SNAP_EPOCHS = (1, 5, 10, 20, 40, 60)
LONG = 128
NOISE = (0.05, 0.1)
TRAIN_REPLAY = 400  # training sequences replayed for train MSE (same count as the test set)

# name -> (label, better, group, positive). `positive` metrics get geometric-mean ratios; the others
# (signed or bounded) get differences.
METRICS = {
    "mse": ("test MSE", "lower", "accuracy", True),
    "rmse": ("test RMSE", "lower", "accuracy", True),
    "mae": ("test MAE", "lower", "accuracy", True),
    "median_ae": ("median |error|", "lower", "accuracy", True),
    "p95_ae": ("95th pct |error|", "lower", "accuracy", True),
    "max_ae": ("max |error|", "lower", "accuracy", True),
    "final_mse": ("final-step MSE", "lower", "accuracy", True),
    "nmse": ("NMSE", "lower", "accuracy", True),
    "r2": ("R²", "higher", "accuracy", False),
    "skill": ("skill vs baseline", "higher", "accuracy", False),
    "pearson_r": ("Pearson r", "higher", "accuracy", False),
    "abs_bias": ("|bias|", "lower", "accuracy", True),
    "err_sd": ("error sd", "lower", "accuracy", True),
    "worst_seq_mse": ("worst-sequence MSE", "lower", "accuracy", True),
    "p90_seq_mse": ("90th pct sequence MSE", "lower", "accuracy", True),
    "direction_acc": ("directional accuracy", "higher", "accuracy", False),
    "latch_acc": ("latch accuracy", "higher", "accuracy", False),
    **{f"val_ep{e}": (f"val MSE @ epoch {e}", "lower", "convergence", True) for e in SNAP_EPOCHS},
    "val_log_auc": ("val log10-MSE AUC", "lower", "convergence", False),
    "reached_shared": ("fraction of seeds reaching shared threshold", "higher", "convergence", False),
    "epochs_to_shared": ("epochs to shared threshold (censored)", "lower", "convergence", False),
    "seconds_to_shared": ("seconds to shared threshold", "lower", "convergence", False),
    "epochs_to_90pct": ("epochs to 90% own improvement", "lower", "convergence", False),
    "best_epoch": ("best epoch", "lower", "convergence", False),
    "late_train_slope": ("train log10-loss slope, last 10 ep", "lower", "convergence", False),
    "val_spikes": ("val spikes (> 2x running min)", "lower", "stability", False),
    "val_roughness": ("val roughness, last 20 ep", "lower", "stability", True),
    "grad_norm_mean": ("mean grad norm", "lower", "stability", True),
    "train_mse": ("train MSE (replay)", "lower", "generalization", True),
    "test_train_ratio": ("test / train MSE", "lower", "generalization", True),
    "val_train_ratio": ("val / train MSE", "lower", "generalization", True),
    "mse_len64": ("MSE at length 64", "lower", "generalization", True),
    f"mse_len{LONG}": (f"MSE at length {LONG}", "lower", "generalization", True),
    "len64_ratio": ("length 64 / 32 MSE", "lower", "generalization", True),
    f"len{LONG}_ratio": (f"length {LONG} / 32 MSE", "lower", "generalization", True),
    f"len{LONG}_late_early": (f"late / early MSE at length {LONG}", "lower", "generalization", True),
    **{f"mse_noise{s}": (f"MSE, input noise sd {s}", "lower", "generalization", True) for s in NOISE},
    **{f"noise{s}_ratio": (f"noisy / clean MSE (sd {s})", "lower", "generalization", True) for s in NOISE},
    "trainable_parameters": ("trainable parameters", "lower", "cost", False),
    "seconds_per_epoch": ("seconds per epoch", "lower", "cost", True),
    "train_seconds": ("total training seconds", "lower", "cost", True),
}


# ---------------------------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------------------------

def discover(runs_dir):
    runs = []
    for complete in sorted(Path(runs_dir).glob("*/seed_*/*/complete.json")):
        run_dir = complete.parent
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        if config.get("model") in MODELS and config.get("task") in TASKS:
            runs.append((config["task"], config["seeds"]["run_seed"], config["model"], run_dir))
    have = {}
    for task, seed, model, _ in runs:
        have.setdefault(seed, set()).add((task, model))
    need = {(t, m) for t in TASKS for m in MODELS}
    seeds = sorted(s for s, got in have.items() if need <= got)
    return [r for r in runs if r[1] in set(seeds)], seeds


# ---------------------------------------------------------------------------------------------
# Metrics from saved artifacts
# ---------------------------------------------------------------------------------------------

def prediction_metrics(run_dir, task):
    p = pd.read_csv(run_dir / "predictions_test.csv")
    p["is_supervised"] = p.is_supervised.astype(str) == "True"
    p = p.sort_values(["sequence_id", "timestep"])
    sup = p[p.is_supervised]
    err = sup.prediction - sup.target
    ae = err.abs()
    seq_mse = (err ** 2).groupby(sup.sequence_id).mean()
    out = {
        "median_ae": ae.median(), "p95_ae": ae.quantile(0.95), "max_ae": ae.max(),
        "pearson_r": np.corrcoef(sup.prediction, sup.target)[0, 1],
        "abs_bias": abs(err.mean()), "bias": err.mean(), "err_sd": err.std(ddof=0),
        "worst_seq_mse": seq_mse.max(), "p90_seq_mse": seq_mse.quantile(0.9),
    }
    # Directional accuracy: sign(pred_t - y_{t-1}) == sign(y_t - y_{t-1}) where the target moves.
    prev = p.groupby("sequence_id").target.shift(1)
    ok = p.is_supervised & prev.notna() & ((p.target - prev).abs() > 1e-9)
    out["direction_acc"] = (np.sign(p.prediction[ok] - prev[ok]) == np.sign(p.target[ok] - prev[ok])).mean() \
        if task != "flip_flop" else np.nan
    held = sup.target != 0
    out["latch_acc"] = (ae[held] < 0.5).mean() if task == "flip_flop" else np.nan
    return out


def timestep_curve(run_dir):
    p = pd.read_csv(run_dir / "predictions_test.csv")
    p = p[p.is_supervised.astype(str) == "True"]
    return ((p.prediction - p.target) ** 2).groupby(p.timestep).mean()


def history_metrics(history, threshold):
    val, epoch = history.val_mse.to_numpy(), history.epoch.to_numpy()
    logv = np.log10(val)
    out = {f"val_ep{e}": val[epoch == e][0] if (epoch == e).any() else np.nan for e in SNAP_EPOCHS}
    out["val_log_auc"] = logv.mean()
    # Seeds that never reach the threshold are censored one epoch past the budget (and at the total
    # training time), so they count as the slowest instead of dropping out of the paired test.
    hit = np.flatnonzero(val <= threshold)
    seconds = history.epoch_seconds.cumsum().to_numpy()
    out["reached_shared"] = float(len(hit) > 0)
    out["epochs_to_shared"] = epoch[hit[0]] if len(hit) else epoch[-1] + 1
    out["seconds_to_shared"] = seconds[hit[0]] if len(hit) else seconds[-1]
    target = logv[0] - 0.9 * (logv[0] - logv.min())
    out["epochs_to_90pct"] = epoch[np.flatnonzero(logv <= target)[0]]
    tail = history.tail(10)
    out["late_train_slope"] = np.polyfit(tail.epoch, np.log10(tail.train_loss), 1)[0]
    out["val_spikes"] = int((val > 2 * np.minimum.accumulate(val)).sum())
    out["val_roughness"] = np.median(np.abs(np.diff(logv[-20:])))
    out["grad_norm_mean"] = history.grad_norm_mean.mean()
    out["seconds_per_epoch"] = history.epoch_seconds.mean()
    return out


# ---------------------------------------------------------------------------------------------
# Replay of the best checkpoint (inference only)
# ---------------------------------------------------------------------------------------------

def _mse(model, dataset, config, inputs=None):
    from q_slstm.datasets.scalar_tasks import ScalarTaskDataset
    from q_slstm.experiments.scalar_tasks import predict_dataset

    if inputs is not None:
        t = dict(dataset.tensors, inputs=inputs)
        dataset = ScalarTaskDataset(t, dataset.sequence_ids, dataset.task, dataset.metadata)
    pred = predict_dataset(model, dataset, config)
    target = dataset.tensors["targets"][..., 0].numpy().astype(np.float64)
    mask = dataset.tensors["loss_mask"][..., 0].numpy().astype(bool)
    se = (pred - target) ** 2
    counts = mask.sum(0)
    per_step = np.where(counts > 0, np.where(mask, se, 0).sum(0) / np.maximum(counts, 1), np.nan)
    return float(se[mask].mean()), per_step


def replay(job):
    import torch

    from q_slstm.datasets.scalar_tasks import generate_dataset
    from q_slstm.experiments import scalar_tasks as st
    from q_slstm.models.factory import match_recurrence

    task, seed, model_name, run_dir = job
    torch.set_num_threads(1)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    # Inference only, so one large batch per split is fine and much faster than the training batch size.
    config = {**config, "device": "cpu", "run_extrapolation": True, "batch_size": 512}
    data = st.make_datasets(config)
    gen = st.generation_config(config)
    data["long"] = generate_dataset(task, config["extrapolation_size"], LONG, config["seeds"]["data"],
                                    "extrapolation", gen)
    model = st.build_model(config)
    model.load_state_dict(torch.load(run_dir / "checkpoints" / "best.pt", map_location="cpu",
                                     weights_only=False)["model_state_dict"])
    match_recurrence(model, config.get("qslstm_recurrence"))

    row = {"task": task, "seed": seed, "model": model_name}
    row["train_mse"], _ = _mse(model, data["train"].select(list(range(TRAIN_REPLAY))), config)
    row["replay_test_mse"], _ = _mse(model, data["test"], config)
    row["mse_len64"], _ = _mse(model, data["extrapolation"], config)
    row[f"mse_len{LONG}"], long_curve = _mse(model, data["long"], config)
    length = config["sequence_length"]
    row[f"len{LONG}_late_early"] = np.nanmean(long_curve[-length:]) / np.nanmean(long_curve[:length])
    x = data["test"].tensors["inputs"]
    rng = torch.Generator().manual_seed(int(seed) % (2 ** 31))
    for s in NOISE:
        row[f"mse_noise{s}"], _ = _mse(model, data["test"], config, x + s * torch.randn(x.shape, generator=rng))
    curve = [{"task": task, "seed": seed, "model": model_name, "timestep": t, "mse": v}
             for t, v in enumerate(long_curve)]
    return row, curve


def run_replay(runs, out, workers, rereplay):
    cache, curve_cache = out / "replay_metrics.csv", out / "replay_timestep_len128.csv"
    if cache.exists() and curve_cache.exists() and not rereplay:
        return pd.read_csv(cache), pd.read_csv(curve_cache)
    print(f"replaying {len(runs)} best checkpoints ({workers} workers, inference only)", flush=True)
    rows, curves = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for k, (row, curve) in enumerate(pool.map(replay, runs), 1):
            rows.append(row)
            curves += curve
            if k % 20 == 0 or k == len(runs):
                print(f"  {k}/{len(runs)}", flush=True)
    rows, curves = pd.DataFrame(rows), pd.DataFrame(curves)
    rows.to_csv(cache, index=False)
    curves.to_csv(curve_cache, index=False)
    return rows, curves


# ---------------------------------------------------------------------------------------------
# Paired statistics
# ---------------------------------------------------------------------------------------------

def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def compare(frame, n_boot=4000):
    rng = np.random.default_rng(0)
    rows = []
    for metric, (label, better, group, positive) in METRICS.items():
        if metric not in frame:
            continue
        for task in TASKS:
            wide = frame[frame.task == task].pivot(index="seed", columns="model", values=metric).dropna()
            if wide.empty:
                continue
            a, b = wide[A].to_numpy(float), wide[B].to_numpy(float)
            sign = 1.0 if better == "lower" else -1.0
            d = b - a
            gain = -sign * d  # > 0 when Q-sLSTM is better
            row = {"task": task, "metric": metric, "label": label, "group": group, "better": better, "n": len(d)}
            for name, v in ((A, a), (B, b)):
                row[f"{name}_mean"], row[f"{name}_sd"], row[f"{name}_median"] = v.mean(), v.std(ddof=1), np.median(v)
            row["qslstm_wins"] = int((gain > 0).sum())
            row["ties"] = int((gain == 0).sum())
            row["mean_diff"] = d.mean()
            nz = gain[gain != 0]
            row["wilcoxon_p"] = stats.wilcoxon(nz).pvalue if len(nz) >= 2 else 1.0
            ranks = stats.rankdata(np.abs(nz))
            row["rank_biserial"] = (ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum() if len(nz) else 0.0
            row["cohens_dz"] = gain.mean() / gain.std(ddof=1) if gain.std(ddof=1) > 0 else np.nan
            if positive and (a > 0).all() and (b > 0).all():
                logr = np.log(b) - np.log(a)
                boots = rng.integers(0, len(logr), (n_boot, len(logr)))
                row["ratio"] = math.exp(logr.mean())
                row["ratio_ci_low"], row["ratio_ci_high"] = np.exp(np.quantile(logr[boots].mean(1), [0.025, 0.975]))
            else:
                row["ratio"] = row["ratio_ci_low"] = row["ratio_ci_high"] = np.nan
            rows.append(row)
    table = pd.DataFrame(rows)
    table["wilcoxon_p_holm"] = table.groupby("metric").wilcoxon_p.transform(lambda p: holm(p.to_numpy()))
    return table


# ---------------------------------------------------------------------------------------------
# Figures
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


def task_grid(title, ncols=3, width=4.4, height=3.5):
    rows = -(-len(TASKS) // ncols)
    fig, axes = plt.subplots(rows, ncols, figsize=(width * ncols, height * rows + 0.8), facecolor=SURFACE,
                             squeeze=False)
    fig.suptitle(title, color=INK, fontsize=13, x=0.01, ha="left")
    for ax in axes.flat:
        style_axes(ax)
    return fig, dict(zip(TASKS, axes.flat))


def save(fig, path, n_seeds, note="", extra=()):
    handles = [plt.Line2D([], [], color=COLORS[m], linewidth=2.5, label=LABELS[m]) for m in MODELS] + list(extra)
    fig.legend(handles=handles, loc="upper right", ncol=len(handles), frameon=False, fontsize=9,
               labelcolor=INK, bbox_to_anchor=(0.99, 0.995))
    fig.text(0.01, 0.005, f"{n_seeds} paired seeds, best-validation checkpoint. {note}".strip(),
             color=MUTED, fontsize=8, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", path)


def short_log_labels(ax):
    """Plain tick labels on a log axis; a range under a decade otherwise gets 2.5x10^-1 minor labels."""
    from matplotlib.ticker import NullFormatter

    short = FuncFormatter(lambda v, _: f"{v:.3g}")
    lo, hi = ax.get_ylim()
    ax.yaxis.set_major_formatter(short)
    ax.yaxis.set_minor_formatter(short if hi / lo < 10 else NullFormatter())
    ax.tick_params(axis="y", which="minor", labelsize=7, colors=MUTED)


def median_band(ax, frame, x, y, model, linestyle="-"):
    g = frame[frame.model == model].groupby(x)[y]
    med = g.median()
    ax.fill_between(med.index, g.quantile(0.25), g.quantile(0.75), color=COLORS[model], alpha=0.18, linewidth=0)
    ax.plot(med.index, med.values, color=COLORS[model], linewidth=2, linestyle=linestyle)


def fig_overview(table, path, n_seeds):
    metrics = [m for m in METRICS if m in set(table.metric) and table[table.metric == m].rank_biserial.abs().sum() > 0]
    grid = np.full((len(metrics), len(TASKS)), np.nan)
    stars = [["" for _ in TASKS] for _ in metrics]
    for i, m in enumerate(metrics):
        for j, t in enumerate(TASKS):
            r = table[(table.metric == m) & (table.task == t)]
            if len(r):
                grid[i, j] = r.rank_biserial.iloc[0]
                p = r.wilcoxon_p_holm.iloc[0]
                stars[i][j] = f"{r.qslstm_wins.iloc[0]}" + ("*" if p < 0.05 else "")
    cmap = LinearSegmentedColormap.from_list("ab", [COLORS[A], "#f2f1ee", COLORS[B]])
    fig, ax = plt.subplots(figsize=(8.6, 0.3 * len(metrics) + 1.8), facecolor=SURFACE)
    im = ax.imshow(grid, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    for i in range(len(metrics)):
        for j in range(len(TASKS)):
            if stars[i][j]:
                ax.text(j, i, stars[i][j], ha="center", va="center", fontsize=7, color=INK)
    ax.set_xticks(range(len(TASKS)), TASKS, fontsize=8, color=INK)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(metrics)), [f"{METRICS[m][0]}  [{METRICS[m][2]}]" for m in metrics], fontsize=7.5,
                  color=INK)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("rank-biserial effect  (− QLSTM better · + Q-sLSTM better)", fontsize=8, color=MUTED)
    cb.ax.tick_params(labelsize=7, colors=MUTED)
    fig.suptitle("QLSTM vs Q-sLSTM: who wins each metric", color=INK, fontsize=12, x=0.01, ha="left")
    fig.text(0.01, 0.005, f"Cell text = seeds (of {n_seeds}) where Q-sLSTM is better; * = Wilcoxon p < 0.05 "
             "after Holm correction over the six tasks.", color=MUTED, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", path)


def fig_paired(frame, metric, path, n_seeds, table):
    fig, panels = task_grid(f"{METRICS[metric][0]} per seed (lines join paired seeds, log scale)")
    for task, ax in panels.items():
        wide = frame[frame.task == task].pivot(index="seed", columns="model", values=metric)
        for _, r in wide.iterrows():
            better = r[B] < r[A]
            ax.plot([0, 1], [r[A], r[B]], color=COLORS[B] if better else COLORS[A], alpha=0.35, linewidth=1)
        for k, m in enumerate(MODELS):
            ax.scatter(np.full(len(wide), k), wide[m], s=14, color=COLORS[m], zorder=3)
            ax.hlines(np.exp(np.log(wide[m]).mean()), k - 0.18, k + 0.18, color=INK, linewidth=2, zorder=4)
        ax.set_yscale("log")
        ax.set_xticks([0, 1], [LABELS[m] for m in MODELS], color=MUTED)
        ax.set_xlim(-0.4, 1.4)
        r = table[(table.metric == metric) & (table.task == task)].iloc[0]
        ax.set_title(f"{task}   ratio {r.ratio:.2f} [{r.ratio_ci_low:.2f}, {r.ratio_ci_high:.2f}]   "
                     f"Q-sLSTM wins {r.qslstm_wins}/{r.n}", color=INK, fontsize=9, loc="left")
    save(fig, path, n_seeds, "Bar = geometric mean; line colour = the model that won that seed; "
         "ratio = Q-sLSTM / QLSTM geometric mean with paired-bootstrap 95% CI.")


def fig_error_cdf(runs, path, n_seeds):
    fig, panels = task_grid("Distribution of absolute test errors (all supervised steps, all seeds)")
    for task, ax in panels.items():
        for m in MODELS:
            ae = np.concatenate([
                (lambda p: (p.prediction - p.target).abs()[p.is_supervised.astype(str) == "True"].to_numpy())(
                    pd.read_csv(d / "predictions_test.csv")) for t, _, mm, d in runs if t == task and mm == m])
            ae = np.sort(np.maximum(ae, 1e-6))
            ax.plot(ae, np.arange(1, len(ae) + 1) / len(ae), color=COLORS[m], linewidth=2)
        ax.set_xscale("log")
        ax.set_title(task, color=INK, fontsize=10, loc="left")
        ax.set_xlabel("|prediction − target|", color=MUTED, fontsize=8)
        ax.set_ylabel("fraction of steps ≤ x", color=MUTED, fontsize=8)
    save(fig, path, n_seeds, "Curves further left = smaller errors.")


def fig_timestep(ts32, ts_long, path, n_seeds):
    fig, axes = plt.subplots(2, len(TASKS), figsize=(3.3 * len(TASKS), 6.6), facecolor=SURFACE, squeeze=False)
    fig.suptitle(f"Test MSE by timestep: training length 32 (top) and length {LONG} (bottom)", color=INK,
                 fontsize=13, x=0.01, ha="left")
    for j, task in enumerate(TASKS):
        for i, frame in enumerate((ts32, ts_long)):
            ax = axes[i, j]
            style_axes(ax)
            sel = frame[(frame.task == task) & (frame.mse > 0)]  # unsupervised warm-up steps have no error
            for m in MODELS:
                median_band(ax, sel, "timestep", "mse", m)
            ax.set_yscale("log")
            short_log_labels(ax)
            if i == 1:
                ax.axvline(31.5, color=MUTED, linewidth=0.8, linestyle=":")
                ax.set_xlabel("timestep", color=MUTED, fontsize=8)
            ax.set_title(task if i == 0 else f"{task}, L={LONG}", color=INK, fontsize=9, loc="left")
    save(fig, path, n_seeds, "Median over seeds, band = IQR. Dotted line = training length.")


def fig_val_curves(history, thresholds, path, n_seeds):
    fig, panels = task_grid("Validation MSE by epoch (log scale)")
    for task, ax in panels.items():
        sel = history[history.task == task]
        for m in MODELS:
            median_band(ax, sel, "epoch", "val_mse", m)
            median_band(ax, sel, "epoch", "train_loss", m, linestyle=":")
        ax.axhline(thresholds[task], color=INK, linewidth=0.9, linestyle="--")
        ax.set_yscale("log")
        ax.set_title(task, color=INK, fontsize=10, loc="left")
        ax.set_xlabel("epoch", color=MUTED, fontsize=8)
    extra = [plt.Line2D([], [], color=MUTED, linewidth=2, label="validation"),
             plt.Line2D([], [], color=MUTED, linewidth=2, linestyle=":", label="train"),
             plt.Line2D([], [], color=INK, linewidth=0.9, linestyle="--", label="shared threshold")]
    save(fig, path, n_seeds, "Median, band = IQR. Shared threshold = the worse model's median best "
         "validation MSE (a level both models typically reach).", extra)


def fig_convergence(frame, path, n_seeds):
    cols = ("epochs_to_shared", "epochs_to_90pct", "val_log_auc")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), facecolor=SURFACE)
    fig.suptitle("Convergence speed", color=INK, fontsize=13, x=0.01, ha="left")
    rng = np.random.default_rng(0)
    for ax, col in zip(axes, cols):
        style_axes(ax)
        for j, task in enumerate(TASKS):
            for k, m in enumerate(MODELS):
                sel = frame[(frame.task == task) & (frame.model == m)]
                v = sel[col].where(sel.reached_shared > 0) if col == "epochs_to_shared" else sel[col]
                reached = v.dropna()
                x = j + (k - 0.5) * 0.36
                ax.scatter(x + rng.uniform(-0.07, 0.07, len(reached)), reached, s=8, color=COLORS[m], alpha=0.5)
                if len(reached):
                    ax.hlines(reached.median(), x - 0.14, x + 0.14, color=COLORS[m], linewidth=2.5)
                if v.isna().any():
                    ax.text(x, reached.max() if len(reached) else 0, f"{v.isna().sum()} n/r", fontsize=6,
                            color=COLORS[m], ha="center", va="bottom")
        ax.set_xticks(range(len(TASKS)), TASKS, fontsize=8, color=MUTED, rotation=20)
        ax.set_title(METRICS[col][0] + " (lower = faster)", color=INK, fontsize=10, loc="left")
    save(fig, path, n_seeds, "Bar = median over seeds that reached the threshold; 'n/r' = seeds that never did "
         "(counted as epoch 61 in the tables).")


def fig_generalization(frame, path, n_seeds):
    conds = [("train_mse", "train"), ("best_val_mse", "val"), ("mse", "test"), ("mse_len64", "L=64"),
             (f"mse_len{LONG}", f"L={LONG}"), *[(f"mse_noise{s}", f"noise {s}") for s in NOISE]]
    conds = [(c, l) for c, l in conds if c in frame]
    fig, panels = task_grid("Generalization: MSE on training data, held-out data, longer sequences and noisy "
                            "inputs (log scale)")
    for task, ax in panels.items():
        sel = frame[frame.task == task]
        for k, m in enumerate(MODELS):
            s = sel[sel.model == m]
            gm = [np.exp(np.log(s[c]).mean()) for c, _ in conds]
            lo = [np.exp(np.log(s[c]).quantile(0.25)) for c, _ in conds]
            hi = [np.exp(np.log(s[c]).quantile(0.75)) for c, _ in conds]
            x = np.arange(len(conds)) + (k - 0.5) * 0.36
            ax.bar(x, gm, width=0.34, color=COLORS[m], alpha=0.85)
            ax.errorbar(x, gm, yerr=[np.subtract(gm, lo), np.subtract(hi, gm)], fmt="none", color=INK,
                        linewidth=1, capsize=2)
        ax.set_yscale("log")
        short_log_labels(ax)
        ax.set_xticks(range(len(conds)), [l for _, l in conds], fontsize=7, color=MUTED, rotation=30)
        ax.set_title(task, color=INK, fontsize=10, loc="left")
    save(fig, path, n_seeds, "Bar = geometric mean over seeds, whiskers = IQR.")


# ---------------------------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------------------------

def fmt_val(v):
    if pd.isna(v):
        return "–"
    return f"{v:.3g}" if abs(v) < 1e4 else f"{v:.2e}"


def write_summary(out, table, runs_dir, n_seeds, thresholds, replayed, mismatch):
    lines = ["# QLSTM vs Q-sLSTM (odd gate) — scalar tasks", "",
             f"Sweep: `{runs_dir.relative_to(ROOT).as_posix()}`, {n_seeds} paired seeds per task, "
             "best-validation checkpoint.", "",
             "Q-sLSTM here is the odd-gate recurrence: gates (1+q)/(1−q) of the VQC expectation q, "
             "xLSTM-stabilized. Both models have 45 trainable parameters.", "",
             "**Reading the tables.** Values are mean over seeds (median in brackets). *ratio* = Q-sLSTM / "
             "QLSTM geometric mean with a paired-bootstrap 95% CI (< 1 = Q-sLSTM lower). *wins* = seeds "
             "where Q-sLSTM is better. *p* = Wilcoxon signed-rank, Holm-adjusted over the six tasks. "
             "*r* = rank-biserial effect, + = Q-sLSTM better.", ""]
    if not replayed:
        lines += ["Generalization replay was skipped (`--no-replay`); replay-based rows are absent.", ""]
    elif mismatch:
        lines += [f"**Warning:** replayed test MSE differs from the recorded value by up to {mismatch:.1%}.", ""]
    else:
        lines += ["Replay check: re-running each best checkpoint on the test set reproduces the recorded "
                  "test MSE (max relative difference < 0.1%).", ""]

    # Scoreboard
    lines += ["## Scoreboard", "", "Metrics (of those reported for the task) where each model is significantly "
              "better (Holm p < 0.05).", "", "| task | Q-sLSTM better | QLSTM better | no significant difference |",
              "|---|---|---|---|"]
    for task in TASKS:
        sel = table[(table.task == task) & (table.group != "cost")]
        sig = sel.wilcoxon_p_holm < 0.05
        lines.append(f"| {task} | {int((sig & (sel.rank_biserial > 0)).sum())} | "
                     f"{int((sig & (sel.rank_biserial < 0)).sum())} | {int((~sig).sum())} |")
    lines.append("")

    lines += ["The scoreboard counts related metrics separately (e.g. MSE, RMSE and NMSE), so read it as a "
              "tally, not independent evidence.", ""]
    lines += ["Shared convergence threshold per task (the worse model's median best validation MSE): "
              + ", ".join(f"{t} {thresholds[t]:.3g}" for t in TASKS) + ".", ""]

    for group in ("accuracy", "convergence", "stability", "generalization", "cost"):
        sel = table[table.group == group]
        if sel.empty:
            continue
        lines += [f"## {group.capitalize()}", ""]
        for metric in [m for m in METRICS if m in set(sel.metric)]:
            ms = sel[sel.metric == metric]
            label, better = METRICS[metric][0], METRICS[metric][1]
            lines += [f"### {label} ({better} is better)", "",
                      "| task | QLSTM | Q-sLSTM | ratio [95% CI] | wins | p | r |", "|---|---|---|---|---|---|---|"]
            for _, r in ms.iterrows():
                ratio = "–" if pd.isna(r.ratio) else f"{r.ratio:.2f} [{r.ratio_ci_low:.2f}, {r.ratio_ci_high:.2f}]"
                lines.append(f"| {r.task} | {fmt_val(r.qlstm_mean)} ({fmt_val(r.qlstm_median)}) | "
                             f"{fmt_val(r.qslstm_mean)} ({fmt_val(r.qslstm_median)}) | {ratio} | "
                             f"{r.qslstm_wins}/{r.n} | {r.wilcoxon_p_holm:.2g} | {r.rank_biserial:+.2f} |")
            lines.append("")

    lines += ["## Figures", ""] + [f"- `{p}`" for p in (
        "fig_overview.png", "fig_test_mse_paired.png", "fig_error_cdf.png", "fig_timestep.png",
        "fig_val_curves.png", "fig_convergence.png", "fig_generalization.png")]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", out / "summary.md")


# ---------------------------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "results/scalar_tasks/paper/2026-09-28")
    parser.add_argument("--out-dir", type=Path, default=None, help=f"default: <runs-dir>/analysis/{OUT_NAME}")
    parser.add_argument("--workers", type=int, default=8, help="processes for the checkpoint replay")
    parser.add_argument("--no-replay", action="store_true", help="skip the replay-based generalization metrics")
    parser.add_argument("--rereplay", action="store_true", help="recompute the cached replay")
    args = parser.parse_args(argv)
    runs_dir = args.runs_dir.resolve()
    out = args.out_dir or runs_dir / "analysis" / OUT_NAME
    out.mkdir(parents=True, exist_ok=True)

    runs, seeds = discover(runs_dir)
    if not seeds:
        raise SystemExit(f"no seed has both {MODELS} on all tasks under {runs_dir}")
    n = len(seeds)
    print(f"{len(runs)} runs, {n} paired seeds", flush=True)

    histories, rows, ts32 = [], [], []
    for task, seed, model, d in runs:
        h = pd.read_csv(d / "history.csv").assign(task=task, seed=seed, model=model)
        histories.append(h)
        ps = pd.read_csv(d / "per_seed_metrics_test.csv").iloc[0]
        done = json.loads((d / "complete.json").read_text(encoding="utf-8"))
        timing = json.loads((d / "timing.json").read_text(encoding="utf-8"))
        rows.append({"task": task, "seed": seed, "model": model,
                     **{k: ps[k] for k in ("mse", "mae", "final_mse", "nmse", "r2", "skill")},
                     "rmse": math.sqrt(ps["mse"]), "best_val_mse": done["best_val_mse"],
                     "best_epoch": done["best_epoch"], "trainable_parameters": done["trainable_parameters"],
                     "train_seconds": timing["train_seconds"], **prediction_metrics(d, task)})
        ts32.append(timestep_curve(d).rename("mse").reset_index().assign(task=task, seed=seed, model=model))
    history = pd.concat(histories, ignore_index=True)
    frame = pd.DataFrame(rows)
    ts32 = pd.concat(ts32, ignore_index=True)

    thresholds = {t: frame[frame.task == t].groupby("model").best_val_mse.median().max() for t in TASKS}
    conv = []
    for (task, seed, model), h in history.groupby(["task", "seed", "model"]):
        conv.append({"task": task, "seed": seed, "model": model,
                     **history_metrics(h.sort_values("epoch"), thresholds[task])})
    frame = frame.merge(pd.DataFrame(conv), on=["task", "seed", "model"])

    replayed, mismatch, ts_long = False, 0.0, None
    if not args.no_replay:
        rep, ts_long = run_replay(runs, out, args.workers, args.rereplay)
        frame = frame.merge(rep, on=["task", "seed", "model"])
        mismatch = float((frame.replay_test_mse / frame.mse - 1).abs().max())
        mismatch = mismatch if mismatch > 1e-3 else 0.0
        frame["test_train_ratio"] = frame.mse / frame.train_mse
        frame["val_train_ratio"] = frame.best_val_mse / frame.train_mse
        frame["len64_ratio"] = frame.mse_len64 / frame.mse
        frame[f"len{LONG}_ratio"] = frame[f"mse_len{LONG}"] / frame.mse
        for s in NOISE:
            frame[f"noise{s}_ratio"] = frame[f"mse_noise{s}"] / frame.mse
        replayed = True

    frame.to_csv(out / "metrics_per_seed.csv", index=False)
    table = compare(frame)
    table.to_csv(out / "comparison.csv", index=False)
    g = history.groupby(["task", "model", "epoch"])[["train_loss", "val_mse"]]
    g.quantile(0.5).join(g.quantile(0.25), rsuffix="_q25").join(g.quantile(0.75), rsuffix="_q75").reset_index() \
        .to_csv(out / "curves_epoch.csv", index=False)
    curves = [ts32.assign(length=32)] + ([ts_long.assign(length=LONG)] if ts_long is not None else [])
    g = pd.concat(curves).groupby(["length", "task", "model", "timestep"]).mse
    pd.DataFrame({"median": g.median(), "q25": g.quantile(0.25), "q75": g.quantile(0.75)}).reset_index() \
        .to_csv(out / "curves_timestep.csv", index=False)

    fig_overview(table, out / "fig_overview.png", n)
    fig_paired(frame, "mse", out / "fig_test_mse_paired.png", n, table)
    fig_error_cdf(runs, out / "fig_error_cdf.png", n)
    if ts_long is not None:
        fig_timestep(ts32, ts_long, out / "fig_timestep.png", n)
    fig_val_curves(history, thresholds, out / "fig_val_curves.png", n)
    fig_convergence(frame, out / "fig_convergence.png", n)
    fig_generalization(frame, out / "fig_generalization.png", n)
    write_summary(out, table, runs_dir, n, thresholds, replayed, mismatch)
    return 0


if __name__ == "__main__":
    sys.exit(main())
