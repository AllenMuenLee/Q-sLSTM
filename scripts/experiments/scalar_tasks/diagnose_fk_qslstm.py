# scripts/experiments/scalar_tasks/diagnose_fk_qslstm.py
#
# Why does fk_qslstm beat QLSTM and Q-sLSTM on the scalar tasks? Replays each best checkpoint on
# test sequences and records, per run:
#   * forget-gate values actually used. QLSTM / Q-sLSTM take sigmoid(q) of a raw Pauli-Z expectation
#     q in [-1, 1], so f is confined to [0.269, 0.731]; fk_qslstm uses exp(Linear(VQC(Linear(.)))).
#   * write proportion alpha = i / (f * n_before + i): the share of memory mass written this step
#     (alpha ~ 0 holds memory, alpha ~ 1 overwrites it); on flip_flop split into pulse vs silent steps.
#   * the range of raw VQC outputs and, for fk_qslstm, of the angles its input encoder feeds the VQC.
#   * a linearization test: each VQC is replaced by the least-squares affine map from its inputs to its
#     outputs (fit on half the test sequences), and test MSE is re-measured on the other half. If the
#     linearized model is as good, the circuit's nonlinearity is not what the model relies on.
# Writes per-run and per-(task, model) tables plus figures to <analysis-dir>/fk_mechanism.

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
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from analyze_three_models import (COLORS, INK, LABELS, MODELS, MUTED, SURFACE, TASKS, discover,  # noqa: E402
                                  paired_seeds, style_axes, task_grid)

GATES = ("input_gate", "forget_gate", "cell_gate", "output_gate")
SIGMOID_RANGE = (1 / (1 + np.e), 1 / (1 + np.e ** -1))  # sigmoid(-1), sigmoid(1)


class Recorder(nn.Module):
    """Wraps a VQC; records per-step (input, output), or replays an affine map instead of the circuit."""

    def __init__(self, vqc):
        super().__init__()
        self.vqc, self.inputs, self.outputs, self.affine, self.record = vqc, [], [], None, True

    def forward(self, x):
        if self.affine is not None:
            weight, bias = self.affine
            return x @ weight + bias
        y = self.vqc(x)
        if self.record:
            self.inputs.append(x.detach())
            self.outputs.append(y.detach())
        return y

    def stacked(self):
        """Recorded inputs and outputs as [batch, steps, features]."""
        return torch.stack(self.inputs, 1), torch.stack(self.outputs, 1)

    def fit(self, n_sequences):
        """Least-squares affine map on the first `n_sequences` recorded sequences; returns mean R^2."""
        x, y = (z[:n_sequences].reshape(-1, z.shape[-1]).double() for z in self.stacked())
        design = torch.cat([x, torch.ones(len(x), 1, dtype=x.dtype)], dim=1)
        coef = torch.linalg.lstsq(design, y).solution
        residual = ((design @ coef - y) ** 2).sum(0)
        r2 = 1 - residual / ((y - y.mean(0)) ** 2).sum(0).clamp_min(1e-30)
        self.affine = (coef[:-1].float(), coef[-1].float())
        return float(r2.mean())


def rollout(model, model_name, inputs):
    """Outputs [B, L] plus per-step forget gates and write proportions [B, L, H]."""
    cell = model.cell
    batch, length, _ = inputs.shape
    n_states = 4 if model_name == "qslstm" else 2
    state = tuple(inputs.new_zeros(batch, cell.hidden_size) for _ in range(n_states))
    n_diag = None
    outs, forgets, alphas = [], [], []
    with torch.no_grad():
        for t in range(length):
            x_t = inputs[:, t]
            combined = torch.cat((x_t, state[0]), dim=-1)
            if model_name == "fk_qslstm":
                cell.forget_gate.record = False  # this extra call must not be recorded twice
                f = torch.exp(cell.Elayer_out_forget(cell.forget_gate(cell.Elayer_in_forget(combined))))
                cell.forget_gate.record = True
                y, *state, diag = cell(x_t, state, return_diagnostics=True, n_diag=n_diag)
                n_diag = diag["n_diag"]
            elif model_name == "qlstm":
                cell.forget_gate.record = False
                f = torch.sigmoid(cell.forget_gate(combined))
                cell.forget_gate.record = True
                y, *state, diag = cell(x_t, state, return_diagnostics=True, n_diag=n_diag)
                n_diag = diag["n_diag"]
            else:  # qslstm: the raw sigmoid forget gate, before stabilizer rescaling
                cell.forget_gate.record = False
                f = torch.sigmoid(cell.forget_gate(combined))
                cell.forget_gate.record = True
                y, *state, diag = cell(x_t, state, return_diagnostics=True)
            outs.append(y[..., 0])
            forgets.append(f)
            alphas.append(diag["alpha"])
    return torch.stack(outs, 1), torch.stack(forgets, 1), torch.stack(alphas, 1)


def masked_mse(pred, targets, mask):
    return float(((pred - targets) ** 2)[mask].mean())


def diagnose(job):
    from plot_curves_and_gates import load_test_data
    from q_slstm.experiments import scalar_tasks as st

    task, seed, model_name, run_dir, n_sequences = job
    torch.set_num_threads(1)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    test = load_test_data(config)
    model = st.build_model(config)
    state = torch.load(run_dir / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    # Current Q-sLSTM cells amplify the forget gate; replay the sigmoid-forget runs this script reads as trained.
    from q_slstm.models.factory import match_recurrence
    match_recurrence(model, config.get("qslstm_recurrence"))
    cell = model.cell

    x = test.tensors["inputs"][:n_sequences]
    y = test.tensors["targets"][:n_sequences, :, 0]
    mask = test.tensors["loss_mask"][:n_sequences, :, 0]
    half = n_sequences // 2

    recorders = {g: Recorder(getattr(cell, g)) for g in GATES}
    for g, rec in recorders.items():
        setattr(cell, g, rec)
    pred, forget, alpha = rollout(model, model_name, x)
    vqc_in = torch.cat([r.stacked()[0].reshape(-1) for r in recorders.values()])
    vqc_out = torch.cat([r.stacked()[1].reshape(-1) for r in recorders.values()])

    row = {"task": task, "seed": seed, "model": model_name,
           "mse": masked_mse(pred[half:], y[half:], mask[half:])}
    f = forget[:, 1:].flatten()
    a = alpha[:, 1:].flatten()  # step 0 writes into an empty memory, alpha = 1 by definition
    for q in (0.05, 0.5, 0.95):
        row[f"forget_p{int(q * 100)}"] = float(f.quantile(q))
        row[f"alpha_p{int(q * 100)}"] = float(a.quantile(q))
    row["forget_max"] = float(f.max())
    row["frac_forget_above_0.9"] = float((f > 0.9).float().mean())
    row["vqc_out_absmax"] = float(vqc_out.abs().max())
    row["frac_vqc_out_above_0.9"] = float((vqc_out.abs() > 0.9).float().mean())
    row["vqc_angle_absmax"] = float(vqc_in.abs().max())
    row["vqc_angle_p95"] = float(vqc_in.abs().quantile(0.95))
    if task == "flip_flop":
        pulse = (x[:, 1:, 0] != 0).unsqueeze(-1).expand_as(alpha[:, 1:])
        row["alpha_pulse"] = float(alpha[:, 1:][pulse].mean())
        row["alpha_silent"] = float(alpha[:, 1:][~pulse].mean())

    # Linearization: fit affine maps on the first half, then replay the second half without circuits.
    row["linear_fit_r2"] = float(np.mean([rec.fit(half) for rec in recorders.values()]))
    lin_pred, _, _ = rollout(model, model_name, x[half:])
    row["mse_linearized"] = masked_mse(lin_pred, y[half:], mask[half:])
    return row


# ---------------------------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------------------------

def finish(fig, out_path, note):
    handles = [plt.Line2D([], [], color=COLORS[m], linewidth=0, marker="o", markersize=7, label=LABELS[m])
               for m in MODELS]
    fig.legend(handles=handles, loc="upper right", ncol=len(MODELS), frameon=False, fontsize=9,
               labelcolor=INK, bbox_to_anchor=(0.99, 0.995))
    fig.text(0.01, 0.005, note, color=MUTED, fontsize=8, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", out_path)


def forget_range_plot(runs, out_path):
    """Per seed: the 5th-95th percentile range of forget-gate values, median as a dot."""
    fig, panels = task_grid("Forget-gate values used on test data (best checkpoint)")
    for task, ax in panels.items():
        sel = runs[runs.task == task]
        for k, model in enumerate(MODELS):
            m = sel[sel.model == model].sort_values("forget_p50").reset_index(drop=True)
            xs = k + np.linspace(-0.3, 0.3, len(m))
            ax.vlines(xs, m.forget_p5, m.forget_p95, color=COLORS[model], alpha=0.5, linewidth=1.5)
            ax.scatter(xs, m.forget_p50, s=12, color=COLORS[model], zorder=3)
        ax.axhspan(*SIGMOID_RANGE, color=MUTED, alpha=0.08, linewidth=0)
        ax.axhline(1.0, color=INK, linewidth=0.8, linestyle=":")
        ax.set_xticks(range(len(MODELS)), [LABELS[m] for m in MODELS], fontsize=8, color=MUTED)
        ax.set_title(task, color=INK, fontsize=10, loc="left")
        ax.set_ylabel("forget gate f", color=MUTED, fontsize=8)
    finish(fig, out_path, "Each line = one seed: 5th-95th percentile of f over test steps and hidden units; dot = "
                          "median. Shaded band = [sigmoid(-1), sigmoid(1)] = [0.269, 0.731], the only values "
                          "QLSTM / Q-sLSTM can reach. Dotted line: f = 1 (perfect retention).")


def linearization_plot(runs, out_path):
    fig, panels = task_grid("Test MSE with each VQC replaced by its best affine fit")
    for task, ax in panels.items():
        sel = runs[runs.task == task]
        lo = min(sel.mse.min(), sel.mse_linearized.min())
        hi = max(sel.mse.max(), sel.mse_linearized.max())
        ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=0.8, linestyle=":")
        for model in MODELS:
            m = sel[sel.model == model]
            ax.scatter(m.mse, m.mse_linearized, s=16, color=COLORS[model], alpha=0.8, zorder=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(task, color=INK, fontsize=10, loc="left")
        ax.set_xlabel("test MSE, trained model", color=MUTED, fontsize=8)
        ax.set_ylabel("test MSE, VQCs linearized", color=MUTED, fontsize=8)
    finish(fig, out_path, "One dot per seed. Affine maps fit on half the test sequences, MSE on the other half. "
                          "On the diagonal: the model does not rely on the circuits' nonlinearity.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "results/scalar_tasks/paper/2026-09-28")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--n-sequences", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    runs_dir = args.runs_dir.resolve()
    out = args.out_dir or runs_dir / "analysis" / "fk_mechanism"
    out.mkdir(parents=True, exist_ok=True)

    runs = discover(runs_dir)
    seeds = set(paired_seeds(runs))
    jobs = [(t, s, m, d, args.n_sequences) for t, s, m, d in runs if s in seeds]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for k, row in enumerate(pool.map(diagnose, jobs), 1):
            rows.append(row)
            if k % 30 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)}", flush=True)
    per_run = pd.DataFrame(rows)
    per_run.to_csv(out / "per_run.csv", index=False)
    summary = per_run.groupby(["task", "model"]).agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary.drop(columns=["seed_mean", "seed_std"]).reset_index().to_csv(out / "summary.csv", index=False)

    forget_range_plot(per_run, out / "forget_gate_range.png")
    linearization_plot(per_run, out / "linearized_vqc_mse.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
