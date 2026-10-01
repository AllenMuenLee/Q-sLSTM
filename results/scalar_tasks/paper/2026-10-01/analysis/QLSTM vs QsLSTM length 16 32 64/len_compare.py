"""QLSTM vs Q-sLSTM across training sequence lengths (16: 2026-09-30, 32: 2026-09-28, 64: 2026-10-01).

Run from the repo root. Train MSE is not saved by the runs, so each best checkpoint is replayed on its
full training split (inference only); results are cached in replay_train_mse.csv. Add a length by adding
an entry to SWEEPS.
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[6]
ROOT = REPO / "results" / "scalar_tasks" / "paper"
SWEEPS = {16: ROOT / "2026-09-30", 32: ROOT / "2026-09-28", 64: ROOT / "2026-10-01"}
OUT = Path(__file__).resolve().parent
MODELS = ["qlstm", "qslstm"]
LABEL = {"qlstm": "QLSTM", "qslstm": "Q-sLSTM"}
COLOR = {"qlstm": "#2a78d6", "qslstm": "#eb6834"}  # same slots as the other scalar-task figures
TASKS = ["delay", "ema", "running_max", "flip_flop", "narma", "sine_next"]
# (length, task, run_seed) dropped from stats and figures. NARMA seed 1106723791 at L=16 has one test sequence
# (id 1000274) whose generated target diverges to ~29 (typical |y| ~0.1); both models score MSE ~181 on it,
# which alone sets the seed's test MSE to 0.46 and the task's SD to 0.10. A generator defect, not model behaviour.
EXCLUDE = {(16, "narma", 1106723791)}
N_NOTE = "20; narma L=16: 19, one diverged test sequence"
EXTRAP_LENGTH = 256  # scripts/experiments/scalar_tasks/extrapolate_checkpoints.py --length
EXTRAP_NOTE = "20 per model and training length"


def replay_train(run_dir):
    sys.path.insert(0, str(REPO / "src"))
    import torch

    from q_slstm.experiments import scalar_tasks as st
    from q_slstm.models.factory import match_recurrence

    torch.set_num_threads(1)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    config = {**config, "device": "cpu", "batch_size": 512}
    data = st.make_datasets(config)
    model = st.build_model(config)
    model.load_state_dict(torch.load(run_dir / "checkpoints" / "best.pt", map_location="cpu",
                                     weights_only=False)["model_state_dict"])
    match_recurrence(model, config.get("qslstm_recurrence"))
    out = {}
    for split in ("train", "test"):
        ds = data[split]
        pred = st.predict_dataset(model, ds, config)
        target = ds.tensors["targets"][..., 0].numpy().astype(np.float64)
        mask = ds.tensors["loss_mask"][..., 0].numpy().astype(bool)
        out[f"{split}_mse"] = float(((pred - target) ** 2)[mask].mean())
    return str(run_dir), out["train_mse"], out["test_mse"]


def load():
    rows, jobs = [], []
    for L, sweep in SWEEPS.items():
        for task in TASKS:
            for seed_dir in sorted((sweep / task).glob("seed_*")):
                for m in MODELS:
                    d = seed_dir / m
                    cfg = json.loads((d / "config.json").read_text())
                    assert cfg["sequence_length"] == L, (d, cfg["sequence_length"])
                    r = pd.read_csv(d / "per_seed_metrics_test.csv").iloc[0].to_dict()
                    r.update(length=L, run_dir=str(d), rmse=np.sqrt(r["mse"]),
                             best_val_mse=json.loads((d / "complete.json").read_text())["best_val_mse"])
                    rows.append(r)
                    jobs.append(d)
    df = pd.DataFrame(rows)

    cache = OUT / "replay_train_mse.csv"
    done = pd.read_csv(cache) if cache.exists() else pd.DataFrame(columns=["run_dir", "train_mse", "replay_test_mse"])
    todo = [j for j in jobs if str(j) not in set(done.run_dir)]
    if todo:
        print(f"replaying {len(todo)} checkpoints on their training split (inference only)", flush=True)
        new = []
        with ProcessPoolExecutor(max_workers=16) as pool:
            for k, res in enumerate(pool.map(replay_train, todo), 1):
                new.append(dict(zip(["run_dir", "train_mse", "replay_test_mse"], res)))
                if k % 40 == 0 or k == len(todo):
                    print(f"  {k}/{len(todo)}", flush=True)
        done = pd.concat([done, pd.DataFrame(new)], ignore_index=True)
        done.to_csv(cache, index=False)
    done[["train_mse", "replay_test_mse"]] = done[["train_mse", "replay_test_mse"]].astype(float)
    df = df.merge(done, on="run_dir", how="left")
    drift = (df.replay_test_mse / df.mse - 1).abs().max()
    assert drift < 1e-3, f"replayed test MSE differs from recorded by {drift:.2%}"
    df["gen_gap"] = df.mse - df.train_mse
    df["excluded"] = [(L, t, s) in EXCLUDE for L, t, s in zip(df.length, df.task, df.run_seed)]
    return df.drop(columns="run_dir")


def pval(a, b):
    try:
        return wilcoxon(a, b).pvalue
    except ValueError:
        return 1.0


def summarize(df, metrics=("mse", "rmse", "mae", "train_mse", "gen_gap")):
    rows = []
    for L in SWEEPS:
        for task in TASKS:
            wide = df[(df.length == L) & (df.task == task)].pivot(index="run_seed", columns="model")
            for met in metrics:
                a, b = wide[met]["qlstm"].to_numpy(), wide[met]["qslstm"].to_numpy()
                rows.append(dict(length=L, task=task, metric=met, n=len(a),
                                 qlstm_mean=a.mean(), qlstm_sd=a.std(ddof=1),
                                 qslstm_mean=b.mean(), qslstm_sd=b.std(ddof=1),
                                 qslstm_lower=int((b < a).sum()), p_wilcoxon=pval(a, b)))
    return pd.DataFrame(rows)


def bar_figure(summary, metric, ylabel, title, fname, zero_line=False, tick="L = {}", note=None):
    lengths = list(SWEEPS)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    w = 0.36
    for ax, task in zip(axes.flat, TASKS):
        s = summary[(summary.task == task) & (summary.metric == metric)].set_index("length").loc[lengths]
        x = np.arange(len(lengths))
        for i, m in enumerate(MODELS):
            ax.bar(x + (i - 0.5) * (w + 0.02), s[f"{m}_mean"], w, yerr=s[f"{m}_sd"], color=COLOR[m],
                   edgecolor="white", linewidth=0, capsize=4, error_kw=dict(lw=1, ecolor="#333333"),
                   label=LABEL[m])
        ax.set_xticks(x, [tick.format(L) for L in lengths])
        ax.set_title(task)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#e5e5e5", lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if zero_line:
            ax.axhline(0, color="#666666", lw=0.8)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, frameon=False)
    fig.suptitle(f"{title} — mean ± SD over seeds ({note or N_NOTE})", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / fname, dpi=150)
    plt.close(fig)


def load_history(df):
    rows = []
    for L, sweep in SWEEPS.items():
        for task in TASKS:
            for seed_dir in sorted((sweep / task).glob("seed_*")):
                seed = int(seed_dir.name.split("_")[1])
                if (L, task, seed) in EXCLUDE:
                    continue
                for m in MODELS:
                    h = pd.read_csv(seed_dir / m / "history.csv", usecols=["epoch", "train_loss", "val_mse"])
                    rows.append(h.assign(length=L, task=task, run_seed=seed, model=m))
    return pd.concat(rows, ignore_index=True)


def loss_figure(hist):
    lengths = list(SWEEPS)
    fig, axes = plt.subplots(len(TASKS), len(lengths), figsize=(13, 19), sharey="row", sharex=True)
    for r, task in enumerate(TASKS):
        for c, L in enumerate(lengths):
            ax = axes[r, c]
            for m in MODELS:
                g = hist[(hist.task == task) & (hist.length == L) & (hist.model == m)].groupby("epoch")
                for col, ls in (("train_loss", "-"), ("val_mse", "--")):
                    mu, sd = g[col].mean(), g[col].std(ddof=1)
                    ax.plot(mu.index, mu, ls, color=COLOR[m], lw=1.4)
                    ax.fill_between(mu.index, (mu - sd).clip(lower=mu.min() * 0.5), mu + sd,
                                    color=COLOR[m], alpha=0.15, lw=0)
            ax.set_yscale("log")
            ax.grid(color="#e5e5e5", lw=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if r == 0:
                ax.set_title(f"L = {L}")
            if c == 0:
                ax.set_ylabel(f"{task}\nMSE (log)")
            if r == len(TASKS) - 1:
                ax.set_xlabel("epoch")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=COLOR[m], lw=2, label=LABEL[m]) for m in MODELS] + [
        Line2D([], [], color="#555555", ls="-", label="train loss"),
        Line2D([], [], color="#555555", ls="--", label="val MSE")]
    fig.legend(handles=handles, loc="upper right", ncol=4, frameon=False)
    fig.suptitle(f"Loss vs epoch — mean ± SD over seeds ({N_NOTE})", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(OUT / "fig_loss_vs_epoch.png", dpi=150)
    plt.close(fig)


def timestep_one(run_dir):
    p = pd.read_csv(run_dir / "predictions_test.csv", usecols=["timestep", "target", "prediction", "is_supervised"])
    p = p[p.is_supervised]
    err = p.prediction - p.target
    g = pd.DataFrame({"timestep": p.timestep, "se": err ** 2, "ae": err.abs()}).groupby("timestep")
    out = pd.DataFrame({"mse": g.se.mean(), "mae": g.ae.mean()}).reset_index()
    out["rmse"] = np.sqrt(out.mse)
    return out


def load_timestep():
    """Per-seed, per-timestep test metrics over supervised steps, from predictions_test.csv (cached)."""
    cache = OUT / "timestep_metrics_per_seed.csv"
    if cache.exists():
        return pd.read_csv(cache)
    keys, dirs = [], []
    for L, sweep in SWEEPS.items():
        for task in TASKS:
            for seed_dir in sorted((sweep / task).glob("seed_*")):
                for m in MODELS:
                    keys.append(dict(length=L, task=task, run_seed=int(seed_dir.name.split("_")[1]), model=m))
                    dirs.append(seed_dir / m)
    with ProcessPoolExecutor(max_workers=16) as pool:
        parts = [r.assign(**k) for k, r in zip(keys, pool.map(timestep_one, dirs))]
    ts = pd.concat(parts, ignore_index=True)
    ts.to_csv(cache, index=False)
    return ts


def timestep_figure(ts, metric, ylabel, extrapolation=False):
    lengths = list(SWEEPS)
    if not extrapolation:
        ts = ts[[(L, t, s) not in EXCLUDE for L, t, s in zip(ts.length, ts.task, ts.run_seed)]]
    fig, axes = plt.subplots(len(TASKS), len(lengths), figsize=(13, 19), sharey="row")
    for r, task in enumerate(TASKS):
        for c, L in enumerate(lengths):
            ax = axes[r, c]
            for m in MODELS:
                g = ts[(ts.task == task) & (ts.length == L) & (ts.model == m)].groupby("timestep")[metric]
                mu, sd = g.mean(), g.std(ddof=1)
                ax.plot(mu.index, mu, color=COLOR[m], lw=1.4, label=LABEL[m])
                ax.fill_between(mu.index, mu - sd, mu + sd, color=COLOR[m], alpha=0.15, lw=0)
            ax.set_xlim(0, (EXTRAP_LENGTH if extrapolation else L) - 1)
            if extrapolation:
                ax.axvline(L - 0.5, color="#666666", lw=0.8, ls=":")
            ax.grid(color="#e5e5e5", lw=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))
            if r == 0:
                ax.set_title(f"trained at L = {L}" if extrapolation else f"L = {L}")
            if c == 0:
                ax.set_ylabel(f"{task}\n{ylabel}")
            if r == len(TASKS) - 1:
                ax.set_xlabel("timestep")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, frameon=False)
    what = (f"Extrapolation ({EXTRAP_LENGTH} steps; dotted line = training length)" if extrapolation else "Test")
    fig.suptitle(f"{what} {ylabel} vs timestep (supervised steps) — mean ± SD over seeds "
                 f"({EXTRAP_NOTE if extrapolation else N_NOTE})", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(OUT / f"fig_{'extrap_' if extrapolation else ''}{metric}_vs_timestep.png", dpi=150)
    plt.close(fig)


def load_extrapolation():
    """Per-seed and per-timestep L=EXTRAP_LENGTH results written by extrapolate_checkpoints.py; None if absent."""
    seeds, steps = [], []
    for L, sweep in SWEEPS.items():
        for task in TASKS:
            for seed_dir in sorted((sweep / task).glob("seed_*")):
                for m in MODELS:
                    d = seed_dir / m / f"extrapolation_L{EXTRAP_LENGTH}"
                    if not (d / "extrapolation.json").exists():
                        return None
                    key = dict(length=L, task=task, model=m)
                    r = pd.read_csv(d / "per_seed_metrics_extrapolation.csv").iloc[0]
                    seeds.append(dict(key, run_seed=int(r.run_seed), mse=r.mse, mae=r.mae, rmse=np.sqrt(r.mse)))
                    ts = pd.read_csv(d / "timestep_metrics_extrapolation.csv")
                    steps.append(ts.assign(rmse=np.sqrt(ts.mse), run_seed=int(r.run_seed), **key))
    return pd.DataFrame(seeds), pd.concat(steps, ignore_index=True)


def main():
    df = load()
    df.to_csv(OUT / "metrics_per_seed.csv", index=False)
    summary = summarize(df[~df.excluded])
    summary.to_csv(OUT / "summary_by_length.csv", index=False)
    bar_figure(summary, "mse", "test MSE", "Test MSE", "fig_mse.png")
    bar_figure(summary, "rmse", "test RMSE", "Test RMSE", "fig_rmse.png")
    bar_figure(summary, "mae", "test MAE", "Test MAE", "fig_mae.png")
    bar_figure(summary, "gen_gap", "test MSE − train MSE", "Generalization gap (test − train MSE)",
               "fig_generalization_gap.png", zero_line=True)
    loss_figure(load_history(df))
    ts = load_timestep()
    for met, lab in (("mse", "MSE"), ("mae", "MAE"), ("rmse", "RMSE")):
        timestep_figure(ts, met, lab)
    extrap = load_extrapolation()
    if extrap is None:
        print(f"no complete L={EXTRAP_LENGTH} extrapolation results; skipping those figures")
        return
    ex_seed, ex_ts = extrap
    ex_seed.to_csv(OUT / f"extrapolation_L{EXTRAP_LENGTH}_per_seed.csv", index=False)
    ex_summary = summarize(ex_seed, metrics=("mse", "rmse", "mae"))
    ex_summary.to_csv(OUT / f"extrapolation_L{EXTRAP_LENGTH}_summary.csv", index=False)
    for met, lab in (("mse", "MSE"), ("mae", "MAE"), ("rmse", "RMSE")):
        bar_figure(ex_summary, met, f"extrapolation {lab}", f"Extrapolation to {EXTRAP_LENGTH} steps: {lab}",
                   f"fig_extrap_{met}.png", tick="trained L = {}", note=EXTRAP_NOTE)
        timestep_figure(ex_ts, met, lab, extrapolation=True)
    print(ex_summary.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    pd.set_option("display.width", 220)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()
