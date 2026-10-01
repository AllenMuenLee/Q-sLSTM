# scripts/experiments/scalar_tasks/plot_epoch_predictions.py
#
# Test-set predictions vs. ground truth at several training epochs, one figure per task and seed:
#   <runs-dir>/figures/predictions_by_epoch/<task>_seed_<seed>.png
# Rows are models, columns are epochs. Each panel overlays the target and the prediction on a few test
# sequences placed end to end; the grey band marks each sequence's unsupervised warm-up steps. The panel
# title gives the test MSE over all test sequences (supervised steps) at that epoch.
#
# Weights come from checkpoints/epoch_NNN.pt, written when training ran with --snapshot-epochs; the final
# epoch falls back to checkpoints/last.pt, so existing runs still plot their last epoch. Panels whose
# epoch has no checkpoint are marked "no checkpoint".

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parents[2] / "src"))

from plot_curves_and_gates import COLORS, INK, LABELS, MUTED, SURFACE, load_test_data, style_axes  # noqa: E402

TARGET_COLOR = INK


def checkpoint_for(run_dir, config, epoch):
    snapshot = run_dir / "checkpoints" / f"epoch_{epoch:03d}.pt"
    if snapshot.exists():
        return snapshot
    last = run_dir / "checkpoints" / "last.pt"
    if epoch == config["epochs"] and last.exists():
        return last
    return None


def predict_at(run_dir, config, test, checkpoint):
    import torch

    from q_slstm.experiments import scalar_tasks as st
    from q_slstm.models.factory import match_recurrence

    model = st.build_model(config)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    match_recurrence(model, config.get("qslstm_recurrence"))
    return st.predict_dataset(model, test, {**config, "device": "cpu"}), state["epoch"]


def plot_task_seed(task, seed, runs, models, epochs, n_sequences, out_path):
    """`runs` maps model -> run_dir for one task and seed."""
    config0 = json.loads((runs[models[0]] / "config.json").read_text(encoding="utf-8"))
    test = load_test_data(config0)
    t = test.tensors
    target = t["targets"][..., 0].numpy().astype(np.float64)
    mask = t["loss_mask"][..., 0].numpy().astype(bool)
    length = target.shape[1]
    shown = slice(0, n_sequences)
    x = np.arange(n_sequences * length)

    fig, axes = plt.subplots(len(models), len(epochs), figsize=(3.6 * len(epochs), 2.6 * len(models) + 0.8),
                             facecolor=SURFACE, squeeze=False, sharey=True)
    fig.suptitle(f"{task}, seed {seed}: test predictions by training epoch "
                 f"(first {n_sequences} test sequences, end to end)",
                 color=INK, fontsize=12, x=0.01, ha="left")
    for r, model in enumerate(models):
        run_dir = runs[model]
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        for c, epoch in enumerate(epochs):
            ax = axes[r, c]
            style_axes(ax)
            for k in range(n_sequences):
                warm = np.flatnonzero(~mask[k])
                if len(warm):
                    ax.axvspan(k * length + warm[0] - 0.5, k * length + warm[-1] + 0.5,
                               color=MUTED, alpha=0.10, linewidth=0)
                if k:
                    ax.axvline(k * length - 0.5, color=MUTED, linewidth=0.6, alpha=0.5)
            ax.plot(x, target[shown].ravel(), color=TARGET_COLOR, linewidth=1.4)
            checkpoint = checkpoint_for(run_dir, config, epoch)
            if epoch > config["epochs"]:
                title = f"epoch {epoch}: run has {config['epochs']} epochs"
            elif checkpoint is None:
                title = f"epoch {epoch}: no checkpoint"
            else:
                pred, _ = predict_at(run_dir, config, test, checkpoint)
                mse = float(((pred - target) ** 2)[mask].mean())
                ax.plot(x, pred[shown].ravel(), color=COLORS[model], linewidth=1.6)
                title = f"epoch {epoch}   test MSE {mse:.4f}"
                print(f"  {task} seed {seed} {model} epoch {epoch}: test MSE {mse:.5f}", flush=True)
            ax.set_title(title, color=INK if checkpoint else MUTED, fontsize=9, loc="left")
            if c == 0:
                ax.set_ylabel(LABELS[model], color=COLORS[model], fontsize=10)
            if r == len(models) - 1:
                ax.set_xlabel("timestep", color=MUTED, fontsize=8)

    handles = [plt.Line2D([], [], color=TARGET_COLOR, linewidth=1.4, label="target")]
    handles += [plt.Line2D([], [], color=COLORS[m], linewidth=2, label=f"{LABELS[m]} prediction") for m in models]
    handles.append(plt.Rectangle((0, 0), 1, 1, color=MUTED, alpha=0.10, label="warm-up (not supervised)"))
    fig.legend(handles=handles, loc="upper right", ncol=len(handles), frameon=False, fontsize=8,
               labelcolor=INK, bbox_to_anchor=(0.99, 0.965))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", out_path, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, required=True,
                        help="a sweep folder, e.g. results/scalar_tasks/paper-overridden/2026-10-01")
    parser.add_argument("--models", nargs="+", default=["qlstm", "qslstm"], choices=sorted(LABELS))
    parser.add_argument("--epochs", type=int, nargs="+", default=[1, 20, 50, 100])
    parser.add_argument("--tasks", nargs="+", default=None, help="default: every task found")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="default: the first seed (sorted) that has every requested model")
    parser.add_argument("--all-seeds", action="store_true", help="one figure for every seed with all models")
    parser.add_argument("--n-sequences", type=int, default=3, help="test sequences shown per panel")
    args = parser.parse_args(argv)

    # (task, seed) -> {model: run_dir}; runs still in training are included, since snapshots exist mid-run.
    found = {}
    for path in sorted(args.runs_dir.rglob("config.json")):
        config = json.loads(path.read_text(encoding="utf-8"))
        if config.get("model") in args.models and (args.tasks is None or config.get("task") in args.tasks):
            found.setdefault((config["task"], config["seeds"]["run_seed"]), {})[config["model"]] = path.parent
    complete = {key: runs for key, runs in found.items() if all(m in runs for m in args.models)}
    if not complete:
        raise SystemExit(f"no task/seed under {args.runs_dir} has all of {args.models}")

    out = args.runs_dir / "figures" / "predictions_by_epoch"
    out.mkdir(parents=True, exist_ok=True)
    for task in sorted({task for task, _ in complete}):
        seeds = sorted(seed for t, seed in complete if t == task)
        if args.seeds is not None:
            seeds = [s for s in seeds if s in args.seeds]
        elif not args.all_seeds:
            seeds = seeds[:1]
        for seed in seeds:
            plot_task_seed(task, seed, complete[(task, seed)], args.models, args.epochs, args.n_sequences,
                           out / f"{task}_seed_{seed}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
