"""QLSTM vs Q-sLSTM across training sequence lengths (32: 2026-09-28, 16: 2026-09-30).

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
SWEEPS = {32: ROOT / "2026-09-28", 16: ROOT / "2026-09-30"}
OUT = Path(__file__).resolve().parent
MODELS = ["qlstm", "qslstm"]
LABEL = {"qlstm": "QLSTM", "qslstm": "Q-sLSTM"}
COLOR = {"qlstm": "#2a78d6", "qslstm": "#eb6834"}  # same slots as the other scalar-task figures
TASKS = ["delay", "ema", "running_max", "flip_flop", "narma", "sine_next"]
# (length, task, run_seed) dropped from stats and figures. NARMA seed 1106723791 at L=16 has one test sequence
# (id 1000274) whose generated target diverges to ~29 (typical |y| ~0.1); both models score MSE ~181 on it,
# which alone sets the seed's test MSE to 0.46 and the task's SD to 0.10. A generator defect, not model behaviour.
EXCLUDE = {(16, "narma", 1106723791)}


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


def summarize(df):
    rows = []
    for L in SWEEPS:
        for task in TASKS:
            wide = df[(df.length == L) & (df.task == task)].pivot(index="run_seed", columns="model")
            for met in ["mse", "rmse", "mae", "train_mse", "gen_gap"]:
                a, b = wide[met]["qlstm"].to_numpy(), wide[met]["qslstm"].to_numpy()
                rows.append(dict(length=L, task=task, metric=met, n=len(a),
                                 qlstm_mean=a.mean(), qlstm_sd=a.std(ddof=1),
                                 qslstm_mean=b.mean(), qslstm_sd=b.std(ddof=1),
                                 qslstm_lower=int((b < a).sum()), p_wilcoxon=pval(a, b)))
    return pd.DataFrame(rows)


def bar_figure(summary, metric, ylabel, title, fname, zero_line=False):
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
        ax.set_xticks(x, [f"L = {L}" for L in lengths])
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
    fig.suptitle(f"{title} — mean ± SD over seeds (20; narma L=16: 19, one diverged test sequence)", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / fname, dpi=150)
    plt.close(fig)


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
    pd.set_option("display.width", 220)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()
