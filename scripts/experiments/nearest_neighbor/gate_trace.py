# scripts/experiments/nearest_neighbor/gate_trace.py
#
# Per-step gate trace of a trained checkpoint: what each gate attempts, and what the memory actually does.
#
#   i_t       input gate   -- attempt to write the new candidate
#   f_t       forget gate  -- attempt to retain the old memory (unbounded above for qslstm)
#   n_{t-1}   normalizer carried in, i.e. the accumulated weight the write competes against
#   k_t       realized move of the normalized memory toward the candidate:
#                 k_t = (c_t/n_t - c_{t-1}/n_{t-1}) / (z_t - c_{t-1}/n_{t-1})
#             which for the qslstm recurrence equals alpha_t = i'_t / (f'_t n_{t-1} + i'_t); both are
#             recorded so the identity doubles as a check on the trace.
#
# qslstm gates are reported twice: raw (exp of the log-gate, the quantity the circuit produces, where
# f > 1 amplifies) and stabilized (i', f' after the xLSTM max-shift, which is what the recurrence uses).
# The stabilizer rescales i, f and n by the same factor, so alpha and k are unaffected by it.
#
# qlstm has no normalizer: its gates are sigmoids and its n is the analysis-only accumulator from
# diagnostics.write_proportion, so its n / alpha / k columns are comparable in form but not in scale.
#
# Steps are labeled event / near_best_distractor / other_distractor exactly as in false_revision.py.

from __future__ import annotations

import argparse
import json
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
import torch  # noqa: E402

import analyze_results as ar  # noqa: E402
from analyze_results import COLORS, MODEL_LABELS, discover_runs  # noqa: E402
from q_slstm.experiments.nearest_neighbor import make_datasets  # noqa: E402
from q_slstm.models.factory import build_quantum_model  # noqa: E402
from q_slstm.models.q_slstm_cell import (  # noqa: E402
    QSLSTM_POLYNOMIAL_RECURRENCE, bounded_log_ratio, polynomial_memory_update, stabilize_gates,
)
from q_slstm.models.q_slstm_log_cell import logarithmic_gate, logarithmic_memory_update  # noqa: E402

GAP_FLOOR = 1e-6  # |z - c/n| below this leaves k undefined rather than dividing by ~0
KINDS = ("event", "near_best_distractor", "other_distractor")


def load_model(run_dir, config):
    model = build_quantum_model(config["model"], config["input_size"], config["hidden_size"],
                                config["output_size"], config["qnn_depth"],
                                gate_epsilon=config["gate_epsilon"])
    state = torch.load(run_dir / "checkpoints" / "best.pt", map_location="cpu")["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


def step_kinds(similarities, event_mask, near_best_delta):
    """event / near_best_distractor / other_distractor per candidate step."""
    running_best = np.maximum.accumulate(similarities, axis=1)
    prev_best = np.concatenate([np.full((len(similarities), 1), -np.inf), running_best[:, :-1]], axis=1)
    near = ~event_mask & (similarities >= prev_best - near_best_delta)
    return np.where(event_mask, "event", np.where(near, "near_best_distractor", "other_distractor"))


def trace_run(run_dir, config, split, n_sequences):
    """One row per (sequence, timestep, hidden unit) with the gate quantities of that step."""
    if config["model"] == "qslstm" and config.get("qslstm_recurrence") == "polynomial_binary_scale_v1":
        raise ValueError("Polynomial v1 did not bound stored states; evaluate these weights in a fresh v2 run before tracing")
    cfg = {**config, "run_extrapolation": split == "extrapolation"}
    ds = make_datasets(cfg)[split]
    model = load_model(run_dir, config)
    cell = model.cell
    qslstm = config["model"] in ar.QSLSTM_MODELS
    log_gates = config["model"] == "qslstm_log"
    polynomial = config["model"] == "qslstm" and config.get("qslstm_recurrence") == QSLSTM_POLYNOMIAL_RECURRENCE
    binary_scale = polynomial or log_gates

    idx = torch.arange(min(n_sequences, len(ds)))
    x = ds.tensors["inputs"][idx]
    B, L, _ = x.shape
    H = config["hidden_size"]
    zeros = torch.zeros(B, H)
    h, c, n, m = zeros, zeros, zeros, zeros
    rec = {k: [] for k in ("i_raw", "f_raw", "i_gate", "f_gate", "n_prev", "n_t", "m_t",
                           "z", "cn_prev", "cn", "alpha", "k", "o", "pred")}
    with torch.no_grad():
        for t in range(L):
            comb = torch.cat((x[:, t], h), dim=-1)
            q_i, q_f = cell.input_gate(comb), cell.forget_gate(comb)
            z = torch.tanh(cell.cell_gate(comb))
            o = torch.sigmoid(cell.output_gate(comb))
            if log_gates:
                c_t, n_t, m_t, i_gate, f_gate = logarithmic_memory_update(
                    q_i, q_f, z, c, n, m, return_forget_weight=True)
                i_raw, f_raw = logarithmic_gate(q_i), logarithmic_gate(q_f)
            elif polynomial:
                c_t, n_t, m_t, i_gate, f_gate = polynomial_memory_update(
                    q_i, q_f, z, c, n, m, return_forget_weight=True)
                i_raw, f_raw = (1 + q_i) / (1 - q_i), (1 + q_f) / (1 - q_f)
            elif qslstm:
                ell_i = bounded_log_ratio(q_i, cell.gate_epsilon)
                ell_f = bounded_log_ratio(q_f, cell.gate_epsilon)
                m_t, i_gate, f_gate = stabilize_gates(ell_i, ell_f, m)
                i_raw, f_raw = torch.exp(ell_i), torch.exp(ell_f)
            else:
                m_t = torch.zeros_like(m)
                i_gate = i_raw = torch.sigmoid(q_i)
                f_gate = f_raw = torch.sigmoid(q_f)
            if not binary_scale:
                c_t = f_gate * c + i_gate * z
                n_t = f_gate * n + i_gate
            previous_denominator = torch.where(n > 0, n, torch.ones_like(n)) if binary_scale else torch.clamp_min(n, GAP_FLOOR)
            denominator = n_t if binary_scale else torch.clamp_min(n_t, GAP_FLOOR)
            cn_prev = c / previous_denominator
            cn = c_t / denominator
            if qslstm:
                h_t = o * cn
            else:
                h_t = o * torch.tanh(c_t)
            gap = z - cn_prev
            k = torch.where(gap.abs() > GAP_FLOOR, (cn - cn_prev) / gap, torch.full_like(gap, float("nan")))
            for name, value in (("i_raw", i_raw), ("f_raw", f_raw), ("i_gate", i_gate), ("f_gate", f_gate),
                                ("n_prev", n), ("n_t", n_t), ("m_t", m_t), ("z", z), ("cn_prev", cn_prev),
                                ("cn", cn), ("alpha", i_gate / denominator),
                                ("k", k), ("o", o)):
                rec[name].append(value.numpy())
            rec["pred"].append(cell.output_post_processing(h_t)[:, 0].numpy()[:, None].repeat(H, axis=1))
            h, c, n, m = h_t, c_t, n_t, m_t

    arrays = {name: np.stack(values, axis=1) for name, values in rec.items()}  # [B, L, H]
    metric = ds.tensors["metric_mask"][idx][..., 0].numpy().astype(bool)
    events = ds.tensors["event_mask"][idx].numpy().astype(bool).reshape(B, L)
    sims = ds.tensors["similarities"][idx].numpy().reshape(B, L)
    kinds = step_kinds(sims, events, config["generation"]["near_best_delta"])

    frame = pd.DataFrame({name: values.reshape(-1) for name, values in arrays.items()})
    frame["scale_kind"] = "binary_exponent" if binary_scale else "log_stabilizer"
    frame["sequence_id"] = np.repeat(ds.sequence_ids[idx].numpy(), L * H)
    frame["case_type"] = np.repeat(np.array(ds.case_types, dtype=object)[idx.numpy()], L * H)
    frame["timestep"] = np.tile(np.repeat(np.arange(L), H), B)
    frame["unit"] = np.tile(np.arange(H), B * L)
    frame["value"] = np.repeat(x[:, :, 2].numpy().reshape(-1), H)
    frame["target"] = np.repeat(ds.tensors["targets"][idx][..., 0].numpy().reshape(-1), H)
    frame["kind"] = np.repeat(kinds.reshape(-1), H)
    frame["is_metric_step"] = np.repeat(metric.reshape(-1), H)
    frame["run_seed"] = config["seeds"]["run_seed"]
    frame["model"] = config["model"]
    frame["split"] = split
    return frame[frame["is_metric_step"]].drop(columns="is_metric_step")


def summarize(frame):
    """Per seed / model / split / kind: the four quantities, plus the gate ratio that sets alpha."""
    df = frame.copy()
    df["f_n_prev"] = df["f_gate"] * df["n_prev"]
    stats = {"i_gate": ["mean", "median"], "f_gate": ["mean", "median"], "i_raw": ["median"],
             "f_raw": ["median"], "n_prev": ["mean", "median"], "n_t": ["mean"], "f_n_prev": ["mean"],
             "alpha": ["mean", "median"], "k": ["mean"], "m_t": ["mean"], "o": ["mean"]}
    out = df.groupby(["run_seed", "model", "split", "kind"]).agg({**stats, "z": "size"})
    out.columns = ["_".join(c).rstrip("_") for c in out.columns]
    return out.rename(columns={"z_size": "n_rows"}).reset_index()


def plot(frame, summary, out_dir):
    # n_prev through the sequence: does the accumulated weight keep growing?
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for model in ar.MODELS:
        sel = frame[frame["model"] == model]
        g = sel.groupby("timestep")["n_prev"]
        axes[0].plot(g.mean().index, g.mean().values, color=COLORS[model], label=MODEL_LABELS[model])
        g2 = sel.groupby("timestep")["alpha"]
        axes[1].plot(g2.mean().index, g2.mean().values, color=COLORS[model], label=MODEL_LABELS[model])
    axes[0].set_ylabel("n_{t-1} (mean over units, seeds)")
    axes[1].set_ylabel("alpha (mean)")
    for ax in axes:
        ax.set_xlabel("timestep")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_title("Accumulated normalizer")
    axes[1].set_title("Write proportion")
    fig.tight_layout()
    fig.savefig(out_dir / "normalizer_and_alpha.png", dpi=150)
    plt.close(fig)

    # gates and realized move, by step kind
    panels = [("i_gate_mean", "input gate i"), ("f_n_prev_mean", "f · n_{t-1}"),
              ("alpha_mean", "alpha"), ("k_mean", "realized k")]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.3 * len(panels), 4))
    x = np.arange(len(KINDS))
    for ax, (column, label) in zip(axes, panels):
        for i, model in enumerate(ar.MODELS):
            sel = summary[summary["model"] == model].groupby("kind")[column]
            mean, std = sel.mean().reindex(KINDS), sel.std(ddof=1).reindex(KINDS)
            ax.bar(x + (i - 0.5) * 0.38, mean, 0.38, yerr=std, capsize=3, color=COLORS[model],
                   alpha=0.8, label=MODEL_LABELS[model])
        ax.set_xticks(x, [k.replace("_", "\n") for k in KINDS], fontsize=8)
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "gates_by_step_kind.png", dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Per-step gate trace of trained nearest-neighbor checkpoints.")
    parser.add_argument("--runs-dir", default="results/nearest_neighbor/paper")
    parser.add_argument("--out-dir", default=None, help="default: <runs-dir>/analysis/gate_trace")
    parser.add_argument("--split", default="test", choices=["test", "extrapolation"])
    parser.add_argument("--n-sequences", type=int, default=64, help="sequences traced per run")
    parser.add_argument("--seeds", type=int, nargs="*", default=None, help="subset of run seeds")
    parser.add_argument("--save-steps", action="store_true", help="also write the full per-step table")
    ar.add_qslstm_model_argument(parser)
    args = parser.parse_args(argv)
    ar.use_qslstm_model(args.qslstm_model)

    runs, _ = discover_runs(args.runs_dir, ar.MODELS)
    if args.seeds:
        runs = {key: v for key, v in runs.items() if key[0] in set(args.seeds)}
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.runs_dir) / ar.analysis_dir_name("analysis") / "gate_trace"
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for (seed, model), (run_dir, config) in sorted(runs.items()):
        frames.append(trace_run(run_dir, config, args.split, args.n_sequences))
        print(f"traced seed {seed} {model}", file=sys.stderr)
    frame = pd.concat(frames, ignore_index=True)
    summary = summarize(frame)
    summary.to_csv(out_dir / f"gate_summary_{args.split}.csv", index=False)
    if args.save_steps:
        frame.to_csv(out_dir / f"gate_steps_{args.split}.csv.gz", index=False, compression="gzip")
    plot(frame, summary, out_dir)

    cols = ["i_gate_mean", "f_gate_mean", "f_n_prev_mean", "n_prev_mean", "alpha_mean", "k_mean"]
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(summary.groupby(["model", "kind"])[cols].agg(["mean", "std"]).T.to_string())
        gap = (summary["alpha_mean"] - summary["k_mean"]).abs().max()
        print(f"\nmax |alpha - k| across runs: {gap:.2e} (identity check; qlstm has no normalized memory)")
    print(f"\nwritten to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
