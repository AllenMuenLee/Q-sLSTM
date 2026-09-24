# scripts/experiments/nearest_neighbor/false_revision.py
#
# Does a model revise selectively, or does it just follow every new candidate value?
#
# `revision_gain` only looks at event steps. Here the same "pull toward the current candidate's value"
# is measured on every step whose predecessor is also a metric step:
#     pull = |p[t-1] - v[t]| - |p[t] - v[t]|        (v = value carried by candidate t)
#     write coefficient k = slope of (p[t] - p[t-1]) on (v[t] - p[t-1])
# On events v[t] is the new target, so pull == revision_gain. On non-events the target is unchanged,
# so any pull toward v[t] is a false revision. An ideal model has k = 1 on events and k = 0 elsewhere.
# Non-events are split into near-best distractors (similarity within near_best_delta of the running
# best) and ordinary distractors.
#
# Candidate values are not stored with the predictions, so each run's test / extrapolation suite is
# regenerated from its saved config (deterministic from the data seed).

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
from scipy import stats  # noqa: E402

import analyze_results as ar  # noqa: E402
from analyze_results import COLORS, MODEL_LABELS, discover_runs  # noqa: E402
from q_slstm.experiments.nearest_neighbor import make_datasets  # noqa: E402

STEP_KINDS = ("event", "near_best_distractor", "other_distractor")
KIND_LABELS = {"event": "true events", "near_best_distractor": "near-best distractors",
               "other_distractor": "other distractors"}


def candidate_frame(config, split):
    """Per-candidate value / similarity / running best for one split, keyed like predictions_*.csv."""
    cfg = {**config, "run_extrapolation": split == "extrapolation"}
    ds = make_datasets(cfg)[split]
    n, length = ds.tensors["inputs"].shape[:2]
    return pd.DataFrame({
        "sequence_id": np.repeat(ds.sequence_ids.numpy(), length),
        "timestep": np.tile(np.arange(length), n),
        "value": ds.tensors["inputs"][:, :, 2].reshape(-1).numpy(),
    })


def step_table(run_dir, config, split, cache):
    preds = pd.read_csv(run_dir / f"predictions_{split}.csv",
                        usecols=["run_seed", "model", "case_type", "sequence_id", "timestep", "is_metric_step",
                                 "is_event", "similarity", "running_best_similarity", "target", "prediction"])
    key = (config["seeds"]["data"], split)
    if key not in cache:
        cache[key] = candidate_frame(config, split)
    df = preds.merge(cache[key], on=["sequence_id", "timestep"], validate="one_to_one")
    df = df.sort_values(["sequence_id", "timestep"])
    g = df.groupby("sequence_id")
    df["prev_pred"] = g["prediction"].shift()
    df["prev_best"] = g["running_best_similarity"].shift()
    df["prev_metric"] = g["is_metric_step"].shift().astype("boolean").fillna(False).astype(bool)
    df = df[df["is_metric_step"] & df["prev_metric"]].copy()

    delta = config["generation"]["near_best_delta"]
    near = ~df["is_event"] & (df["similarity"] >= df["prev_best"] - delta)
    df["kind"] = np.where(df["is_event"], "event", np.where(near, "near_best_distractor", "other_distractor"))
    df["pull"] = (df["prev_pred"] - df["value"]).abs() - (df["prediction"] - df["value"]).abs()
    df["dp"] = df["prediction"] - df["prev_pred"]
    df["offset"] = df["value"] - df["prev_pred"]
    return df


def summarize(df):
    rows = []
    for (seed, model, split, kind), g in df.groupby(["run_seed", "model", "split", "kind"]):
        slope = np.polyfit(g["offset"], g["dp"], 1)[0] if len(g) > 2 else np.nan
        rows.append({"run_seed": seed, "model": model, "split": split, "kind": kind, "n_steps": len(g),
                     "pull": g["pull"].mean(), "write_k": slope})
    return pd.DataFrame(rows)


def paired(summary):
    rows = []
    for (split, kind), g in summary.groupby(["split", "kind"]):
        for metric in ("pull", "write_k"):
            w = g.pivot(index="run_seed", columns="model", values=metric).dropna()
            q = ar.QSLSTM
            d = w[q] - w["qlstm"]
            rows.append({"split": split, "kind": kind, "metric": metric, "n_pairs": len(d),
                         "qlstm_mean": w["qlstm"].mean(), "qlstm_std": w["qlstm"].std(ddof=1),
                         f"{q}_mean": w[q].mean(), f"{q}_std": w[q].std(ddof=1),
                         "diff_mean": d.mean(), f"{q}_higher_seeds": int((d > 0).sum()),
                         "wilcoxon_p": stats.wilcoxon(d).pvalue if len(d) > 1 else np.nan})
    return pd.DataFrame(rows)


def selectivity(summary):
    """Event write coefficient minus distractor write coefficient, per seed."""
    w = summary.pivot_table(index=["run_seed", "model", "split"], columns="kind", values="write_k").reset_index()
    for kind in STEP_KINDS[1:]:
        w[f"selectivity_vs_{kind}"] = w["event"] - w[kind]
    return w


def plot(summary, path):
    splits = list(dict.fromkeys(summary["split"]))
    fig, axes = plt.subplots(len(splits), 2, figsize=(11, 3.9 * len(splits)), squeeze=False)
    width = 0.38
    x = np.arange(len(STEP_KINDS))
    for r, split in enumerate(splits):
        for c, (metric, ylabel) in enumerate((("pull", "pull toward candidate value"),
                                              ("write_k", "write coefficient k"))):
            ax = axes[r, c]
            for i, model in enumerate(ar.MODELS):
                sel = summary[(summary["split"] == split) & (summary["model"] == model)]
                g = sel.groupby("kind")[metric]
                mean, std = g.mean().reindex(STEP_KINDS), g.std(ddof=1).reindex(STEP_KINDS)
                ax.bar(x + (i - 0.5) * width, mean, width, yerr=std, capsize=3, color=COLORS[model],
                       alpha=0.8, label=MODEL_LABELS[model])
            if metric == "write_k":
                ax.axhline(1, color="green", ls="--", lw=0.8)
            ax.axhline(0, color="black", lw=0.8)
            ax.set_xticks(x, [KIND_LABELS[k] for k in STEP_KINDS])
            ax.set_ylabel(f"{ylabel} (mean ± std)")
            ax.set_title(f"{split}: {ylabel}")
            ax.grid(axis="y", alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description="True vs false revision on the nearest-neighbor task.")
    parser.add_argument("--runs-dir", default="results/nearest_neighbor/paper")
    parser.add_argument("--out-dir", default=None, help="default: <runs-dir>/analysis/false_revision")
    parser.add_argument("--splits", nargs="+", default=["test", "extrapolation"])
    ar.add_qslstm_model_argument(parser)
    args = parser.parse_args(argv)
    ar.use_qslstm_model(args.qslstm_model)

    runs, _ = discover_runs(args.runs_dir, ar.MODELS)
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.runs_dir) / ar.analysis_dir_name("analysis") / "false_revision"
    out_dir.mkdir(parents=True, exist_ok=True)

    cache, frames = {}, []
    for (seed, model), (run_dir, config) in sorted(runs.items()):
        for split in args.splits:
            if (run_dir / f"predictions_{split}.csv").exists():
                frames.append(step_table(run_dir, config, split, cache).assign(split=split))
        print(f"loaded seed {seed} {model}", file=sys.stderr)
    summary = summarize(pd.concat(frames, ignore_index=True))
    summary.to_csv(out_dir / "per_seed_revision.csv", index=False)
    table = paired(summary)
    table.to_csv(out_dir / "paired_revision.csv", index=False)
    sel = selectivity(summary)
    sel.to_csv(out_dir / "selectivity.csv", index=False)
    plot(summary, out_dir / "true_vs_false_revision.png")

    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(table.to_string(index=False))
        print()
        cols = [c for c in sel.columns if c.startswith("selectivity") or c in STEP_KINDS]
        print(sel.groupby(["split", "model"])[cols].agg(["mean", "std"]).T.to_string())
    print(f"\nwritten to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
