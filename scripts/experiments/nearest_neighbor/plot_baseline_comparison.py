# scripts/experiments/nearest_neighbor/plot_baseline_comparison.py
#
# Quantum models (QLSTM, Q-sLSTM, Q-sLSTM-log) next to the classical LSTM / sLSTM baselines on the same seeds
# and data, with the constant predictor (mean training target) as the reference line in every panel.
# Validation MSE is the best-epoch value; test MSE is the best-validation checkpoint, overall and per case type.
# Classical rows come from classical_baseline.py (lr 1e-2 by default); only configurations finished on every
# seed are drawn.

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

from q_slstm.models.factory import build_quantum_model  # noqa: E402

# Colour encodes the family; marker shape repeats it so identity never rests on colour alone.
FAMILY = {"quantum": ("#2a78d6", "o"), "LSTM": ("#eb6834", "s"), "sLSTM": ("#1baf7a", "D")}
QUANTUM = {"qlstm": "QLSTM", "qslstm": "Q-sLSTM", "qslstm_log": "Q-sLSTM-log", "fk_qslstm": "Q-sLSTM (fk)"}
CLASSICAL = {"lstm": ("LSTM", "LSTM"), "slstm_sig": ("sLSTM σ-forget", "sLSTM"), "slstm_exp": ("sLSTM exp-forget", "sLSTM")}
PANELS = {"val_mse": "validation MSE", "test_all": "test MSE (all cases)", "test_iid": "test: iid",
          "test_early": "test: early", "test_late": "test: late", "test_near_best": "test: near_best"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"


def quantum_rows(runs_dirs, seeds):
    rows = []
    for model, label in QUANTUM.items():
        run_root = runs_dirs.get(model)
        if run_root is None:
            continue
        for seed in seeds:
            run = run_root / f"seed_{seed}" / model
            config = json.loads((run / "config.json").read_text())
            params = sum(p.numel() for p in build_quantum_model(
                model, config["input_size"], config["hidden_size"], config["output_size"], config["qnn_depth"],
                gate_epsilon=config["gate_epsilon"]).parameters())
            test = pd.read_csv(run / "per_seed_metrics_test.csv").set_index("case_type")["mse"]
            rows.append({"label": label, "family": "quantum", "n_params": params, "run_seed": seed,
                         "val_mse": pd.read_csv(run / "history.csv")["val_mse"].min(),
                         **{f"test_{case}": test[case] for case in ("all", "iid", "early", "late", "near_best")}})
    return pd.DataFrame(rows)


def classical_rows(csvs, lr, seeds):
    table = pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True)
    constant = table[table["model"] == "constant"].drop_duplicates("run_seed")
    net = table[(table["model"] != "constant") & np.isclose(table["lr"], lr) & table["run_seed"].isin(seeds)].copy()
    net = net.drop_duplicates(["model", "run_seed"])
    complete = net.groupby("model")["run_seed"].nunique() == len(seeds)
    net = net[net["model"].isin(complete[complete].index)]
    net["family"] = net["arch"].fillna("lstm").map(lambda a: CLASSICAL[a][1])
    net["label"] = [f"{CLASSICAL[a][0]} h={int(h)}" for a, h in zip(net["arch"].fillna("lstm"), net["hidden"])]
    return net, constant


def plot(frame, constant, path, seeds):
    order = (frame.groupby("label", sort=False).agg(family=("family", "first"), n=("n_params", "first"))
             .reset_index())
    rank = {"quantum": 0, "LSTM": 1, "sLSTM": 1}
    order = order.assign(r=order["family"].map(rank)).sort_values(["r", "n", "label"], kind="stable")
    labels = list(order["label"])
    y = {label: i for i, label in enumerate(labels)}

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6), sharey=True)
    for ax, (column, title) in zip(axes.flat, PANELS.items()):
        base = constant["val_mse" if column == "val_mse" else column].mean()
        ax.axvline(base, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
        ax.text(base, -0.9, f"constant\n{base:.4f}", ha="center", va="bottom", fontsize=7.5, color=MUTED)
        for label in labels:
            sel = frame[frame["label"] == label]
            color, marker = FAMILY[sel["family"].iloc[0]]
            ax.scatter(sel[column], np.full(len(sel), y[label]), s=22, facecolor="white", edgecolor=color,
                       linewidth=1.2, marker=marker, zorder=2)
            mean = sel[column].mean()
            ax.scatter([mean], [y[label]], s=70, color=color, marker=marker, edgecolor="white", linewidth=1.5,
                       zorder=3)
            ax.text(mean, y[label] + 0.32, f"{mean:.4f}", ha="center", va="top", fontsize=7, color=INK)
        ax.set_title(title, fontsize=10, color=INK, loc="left", pad=22)  # clear of the constant label
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)
    params = order.set_index("label")["n"]
    for ax in axes[:, 0]:
        ax.set_yticks(range(len(labels)), [f"{l}  ({int(params[l]):,} params)" for l in labels], color=INK)
        ax.set_ylim(len(labels) - 0.4, -1.1)
    handles = [plt.Line2D([], [], ls="", marker=m, color=c, markersize=8, label=f)
               for f, (c, m) in FAMILY.items()]
    handles += [plt.Line2D([], [], ls="", marker="o", markerfacecolor="white", markeredgecolor=MUTED,
                           markersize=5, label="one seed"),
                plt.Line2D([], [], color=MUTED, ls=(0, (4, 3)), label="constant predictor")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=9)
    fig.suptitle(f"MSE by model, lower is better (mean of {len(seeds)} paired seeds; small markers = seeds; "
                 "classical lr 1e-2)", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    paper = ROOT / "results/nearest_neighbor/paper"
    parser.add_argument("--classical-dir", type=Path, default=paper / "analysis/classical_baseline")
    parser.add_argument("--qlstm-dir", type=Path, default=paper / "2026-09-21 QLSTM vs QsLSTM")
    parser.add_argument("--qslstm-dir", type=Path, default=paper / "2026-09-21 QLSTM vs QsLSTM")
    parser.add_argument("--qslstm-log-dir", type=Path, default=paper / "2026-09-24")
    parser.add_argument("--fk-qslstm-dir", type=Path, default=None,
                        help="sweep with fk_qslstm runs (omitted from the plot when not given)")
    parser.add_argument("--lr", type=float, default=1e-2)
    args = parser.parse_args(argv)

    csvs = sorted(args.classical_dir.rglob("classical_baseline_per_seed.csv"))
    seeds = sorted(pd.read_csv(csvs[0]).query("model == 'constant'")["run_seed"].unique())
    classical, constant = classical_rows(csvs, args.lr, seeds)
    quantum = quantum_rows({"qlstm": args.qlstm_dir, "qslstm": args.qslstm_dir, "qslstm_log": args.qslstm_log_dir,
                            "fk_qslstm": args.fk_qslstm_dir}, seeds)
    frame = pd.concat([quantum, classical], ignore_index=True)
    columns = ["label", "family", "n_params", "run_seed", *PANELS]
    frame[columns].to_csv(args.classical_dir / "comparison_per_seed.csv", index=False)
    summary = frame.groupby("label", sort=False)[list(PANELS)].mean()
    summary.loc["constant"] = constant.rename(columns={"val_mse": "val_mse"})[list(PANELS)].mean()
    summary.to_csv(args.classical_dir / "comparison_summary.csv")
    plot(frame, constant, args.classical_dir / "comparison.png", seeds)
    with pd.option_context("display.width", 200):
        print(summary.round(4).to_string())
    print(f"written to {args.classical_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
