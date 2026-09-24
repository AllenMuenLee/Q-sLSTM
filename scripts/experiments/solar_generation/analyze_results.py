# scripts/experiments/solar_generation/analyze_results.py
#
# Combine the runs of ONE study (both models, paired seeds) into seed tables, the paired
# Q-sLSTM - QLSTM comparison, persistence references, plots, and a machine-generated summary.md.
# Only verified complete runs are used; a pair is accepted only when every pairing check passes.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402  (import before torch: avoids a Windows pyarrow/torch DLL crash)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from q_slstm.experiments import solar_generation_metrics as sm  # noqa: E402
from q_slstm.experiments.solar_generation import run_status, verify_pairing  # noqa: E402

MODELS = ("qlstm", "qslstm")
MODEL_LABELS = {"qlstm": "QLSTM (conventional)", "qslstm": "Q-sLSTM (stabilized)"}
COLORS = {"qlstm": "#4C72B0", "qslstm": "#DD8452", "persistence": "#555555", "truth": "#000000"}
COMPARISONS = [  # (subset, group_type, group, metric)
    ("all_eligible", "overall", "all", "mse"),
    ("all_eligible", "overall", "all", "mae"),
    ("all_eligible", "overall", "all", "rmse"),
    ("all_eligible", "daylight_proxy", "True", "mse"),
    ("all_eligible", "daylight_proxy", "False", "mse"),
    ("all_eligible", "large_ramp", "True", "mse"),
    ("baseline_common", "overall", "all", "mse"),
]


# ---------------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------------

def discover_runs(runs_dir):
    """Verified complete runs {(seed, model): (dir, config)} plus incomplete/failed run directories."""
    runs, not_complete = {}, []
    for cfg_path in sorted(Path(runs_dir).rglob("config.json")):
        config = json.loads(cfg_path.read_text(encoding="utf-8"))
        if config.get("kind") != "solar_generation_run":
            continue
        run_dir = cfg_path.parent
        status = run_status(config, run_dir)
        if status != "complete":
            reason = "failed" if (run_dir / "failure.json").exists() else status
            not_complete.append((run_dir, config, reason))
            continue
        runs[(config["seeds"]["run_seed"], config["model"])] = (run_dir, config)
    studies = {cfg["study_id"] for _, cfg in runs.values()} | {cfg["study_id"] for _, cfg, _ in not_complete}
    if len(studies) > 1:
        raise SystemExit(f"runs of several studies under {runs_dir}: {sorted(studies)}; analyze one study directory")
    return runs, not_complete


def load_table(runs, keys, filename):
    frames = [pd.read_csv(runs[k][0] / filename) for k in keys if (runs[k][0] / filename).exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def requested_seeds(runs_dir):
    out = {}
    for model in MODELS:
        path = Path(runs_dir) / f"seeds_{model}.json"
        out[model] = json.loads(path.read_text(encoding="utf-8"))["seeds"] if path.exists() else None
    return out


# ---------------------------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------------------------

def _finish(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _overall(per_seed, forecaster="model", subset="all_eligible"):
    return per_seed[(per_seed["subset"] == subset) & (per_seed["group_type"] == "overall")
                    & (per_seed["forecaster"] == forecaster)]


def plot_seed_mse(per_seed, path):
    sel = _overall(per_seed)
    pers = _overall(per_seed, "persistence")
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for k, model in enumerate(MODELS):
        vals = sel[sel["model"] == model].sort_values("run_seed")["mse"].to_numpy()
        ax.bar(k, vals.mean() if len(vals) else np.nan, 0.6, color=COLORS[model], alpha=0.55)
        ax.scatter(np.full(len(vals), k), vals, color=COLORS[model], edgecolor="k", s=24, zorder=3)
    if len(pers):
        ax.axhline(pers["mse"].iloc[0], color=COLORS["persistence"], ls="--", lw=1, label="hourly persistence")
        ax.legend(fontsize=8)
    ax.set_xticks(range(len(MODELS)), [MODEL_LABELS[m] for m in MODELS])
    ax.set_ylabel("test MSE (MWh$^2$)")
    ax.set_title("Held-out next-hour MSE; points = seeds")
    _finish(fig, path)


def plot_paired_differences(table, path):
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    if len(table):
        ax.scatter(range(len(table)), table["difference"], color="k", zorder=3)
        ax.axhline(table["difference"].mean(), color="crimson", lw=1.5, label="mean over seeds")
        ax.set_xticks(range(len(table)), [str(s) for s in table["run_seed"]])
        ax.legend(fontsize=8)
    ax.axhline(0, color="gray", ls=":")
    ax.set_xlabel("run seed")
    ax.set_ylabel("MSE(Q-sLSTM) - MSE(QLSTM)  (MWh$^2$)")
    ax.set_title("Paired seed-level differences (negative favors Q-sLSTM)", fontsize=10)
    _finish(fig, path)


def plot_learning_curves(histories, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, col, title in zip(axes, ("train_mse_mwh2", "val_mse_mwh2"), ("training", "validation")):
        for model in MODELS:
            for i, (_, h) in enumerate(histories[histories["model"] == model].groupby("run_seed")):
                ax.plot(h["epoch"], h[col], color=COLORS[model], alpha=0.8, lw=1.2,
                        label=MODEL_LABELS[model] if i == 0 else None)
        ax.set_xlabel("epoch")
        ax.set_title(f"{title} MSE (final step)")
    axes[0].set_ylabel("MSE (MWh$^2$)")
    axes[0].legend(fontsize=8, loc="upper right")
    _finish(fig, path)


def plot_group_error(per_seed, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [3, 1.3]})
    for ax, group_type in zip(axes, ("month", "daylight_proxy")):
        sel = per_seed[(per_seed["subset"] == "all_eligible") & (per_seed["group_type"] == group_type)]
        groups = sorted(sel["group"].astype(str).unique(), key=lambda g: (not g.isdigit(), int(g) if g.isdigit() else g))
        width = 0.27
        series = [(m, sel[(sel["forecaster"] == "model") & (sel["model"] == m)]) for m in MODELS]
        # persistence does not depend on the model; take it from one model's rows
        series.append(("persistence", sel[(sel["forecaster"] == "persistence") & (sel["model"] == MODELS[0])]))
        for k, (name, s) in enumerate(series):
            means = [s[s["group"].astype(str) == g]["mse"].mean() for g in groups]
            ax.bar(np.arange(len(groups)) + (k - 1) * width, means, width, color=COLORS[name], alpha=0.7,
                   label=MODEL_LABELS.get(name, "hourly persistence"))
        ax.set_xticks(range(len(groups)), groups)
        ax.set_title({"month": "test MSE by target month (mean over seeds)",
                      "daylight_proxy": "by daylight proxy (GHI > 20)"}[group_type], fontsize=10)
    axes[0].set_ylabel("MSE (MWh$^2$)")
    axes[0].legend(fontsize=8)
    _finish(fig, path)


def plot_trace(preds, block, seed, path):
    ids = block["window_ids"]
    fig, ax = plt.subplots(figsize=(11, 3.8))
    sel = preds[(preds["run_seed"] == seed) & preds["window_id"].isin(ids)]
    base = sel[sel["model"] == MODELS[0]].sort_values("window_id")
    t = pd.to_datetime(base["target_timestamp_utc"])
    ax.plot(t, base["target_mwh"], color=COLORS["truth"], lw=2, label="observed")
    ax.plot(t, base["persistence_mwh"], color=COLORS["persistence"], ls="--", lw=1, label="hourly persistence")
    for model in MODELS:
        m = sel[sel["model"] == model].sort_values("window_id")
        ax.plot(pd.to_datetime(m["target_timestamp_utc"]), m["prediction_mwh"], color=COLORS[model], lw=1.4,
                label=MODEL_LABELS[model])
    ax.set_ylabel("solar generation (MWh)")
    ax.set_xlabel("target interval end (UTC)")
    kind = "first complete 7-day test block" if block["complete_7_day_block"] else "first contiguous test run (no complete 7-day block)"
    ax.set_title(f"{kind}: {block['first_target_timestamp_utc']} .. {block['last_target_timestamp_utc']}, seed {seed}",
                 fontsize=9)
    ax.legend(fontsize=8, ncol=4)
    _finish(fig, path)


def plot_alpha(alpha, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, group in zip(axes, ("daylight", "night")):
        for model in MODELS:
            sel = alpha[(alpha["model"] == model) & (alpha["target_group"] == group)]
            curve = sel.groupby("input_step")["alpha_mean"].mean()
            ax.plot(curve.index, curve.values, color=COLORS[model], marker="o", ms=3, label=MODEL_LABELS[model])
        ax.set_xlabel("input step (0 excluded; last = forecast origin)")
        ax.set_title(f"{group} targets")
    axes[0].set_ylabel("write proportion alpha (mean over seeds)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Exploratory: write proportion by input position", y=1.02, fontsize=10)
    _finish(fig, path)


# ---------------------------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------------------------

def _fmt(x, digits=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.{digits}f}" if isinstance(x, (float, np.floating)) else str(x)


def _md(df, digits=2):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.astype(object).iterrows():
        lines.append("| " + " | ".join(_fmt(v, digits) for v in row) + " |")
    return "\n".join(lines)


def write_summary(path, ctx):
    s, dm = ctx["study"], ctx["dataset_manifest"]
    q = dm["quality_report"]
    cov = pd.DataFrame(q["coverage_by_split"]).T.reset_index().rename(columns={"index": "split"})[
        ["split", "expected_rows", "valid_rows", "valid_row_fraction", "eligible_windows", "valid_window_fraction"]]
    prim = ctx["primary_summary"]
    pilot_note = ("**This is a pilot study: a pipeline and cost check, not evidence about forecast performance.**"
                  if s["scale_preset"] == "pilot" else "")
    leaning = "Q-sLSTM" if prim["mean_difference"] < 0 else "QLSTM"
    if prim["n_paired_seeds"] < 2:
        verdict = "inconclusive (fewer than two paired seeds)"
    elif prim["ci95_low"] <= 0 <= prim["ci95_high"]:
        verdict = f"inconclusive (95% CI includes 0; mean difference leans toward {leaning})"
    else:
        verdict = f"{leaning} has lower MSE across seeds (95% CI excludes 0)"
    lines = [
        f"# Ontario solar-generation forecasting: study `{ctx['study_id']}` (scale `{s['scale_label']}`)",
        "",
        "Machine-generated by `analyze_results.py`; do not edit by hand.",
        "",
        pilot_note,
        "",
        "## Setup",
        f"- Dataset `{s['dataset_id']}`: IESO SOLAR output (MWh, registered generators >= 20 MW) and Open-Meteo "
        f"ERA5 weather, equal-weight mean of {len(dm['identity']['locations'])} proxy locations; source dates "
        f"{dm['identity']['preparation']['start_date']} .. {dm['identity']['preparation']['end_date']} (EST).",
        f"- Retrospective benchmark with reanalysis weather and revised generation; publication delays are not modeled.",
        f"- Task: {s['sequence_length']} hourly rows x 13 features -> generation in the next hour; chronological "
        f"70/10/20 split (boundaries in the dataset manifest).",
        f"- Model: Linear(13, {s['projection_size']}) + tanh -> {s['n_vqcs_per_cell']} VQCs per cell, "
        f"{s['n_qubits']} qubits, depth {s['qnn_depth']}, hidden {s['hidden_size']}; Adam lr {s['lr']}, "
        f"batch {s['batch_size']}, {s['epochs']} epochs (no early stopping), weight decay {s['weight_decay']}, "
        f"grad clip {s['grad_clip'] or 'off'}.",
        f"- Overrides: training `{json.dumps(s['training_overrides'])}`, dataset `{json.dumps(s['dataset_overrides'])}`.",
        f"- Seeds requested: qlstm {ctx['requested']['qlstm']}, qslstm {ctx['requested']['qslstm']}; "
        f"completed and accepted pairs: {prim['n_paired_seeds']}.",
        f"- Source revision: {ctx['revision']}.",
        "",
        "## Data coverage",
        _md(cov, 4),
        f"- Generation status counts: `{json.dumps(q['generation_status_counts'])}`; invalid-row reasons: "
        f"`{json.dumps(q['invalid_reason_counts'])}`.",
        f"- Interval alignment audit (best weather lag by season, 0 expected): "
        f"`{json.dumps({k: v['best_lag'] for k, v in q['alignment_diagnostic'].items()})}`.",
        "",
        "## Primary outcome: held-out next-hour MSE (MWh^2)",
        f"- Paired Q-sLSTM - QLSTM: mean {_fmt(prim['mean_difference'])} (sd {_fmt(prim['sd_difference'])}), "
        f"95% CI [{_fmt(prim['ci95_low'])}, {_fmt(prim['ci95_high'])}] ({prim['ci_method']}); "
        f"Q-sLSTM lower in {prim['seeds_a_lower']} of {prim['n_paired_seeds']} seeds. Reading: {verdict}.",
        "",
        _md(ctx["primary_table"]),
        "",
        "## Mean and sample sd across seeds (all eligible test targets)",
        _md(ctx["model_summary"]),
        "",
        "## Persistence references on the common subset (daily persistence available)",
        _md(ctx["baseline_table"]),
        "",
        "## Paired differences (Q-sLSTM - QLSTM)",
        _md(ctx["paired"][["subset", "group_type", "group", "metric", "n_paired_seeds", "mean_difference",
                           "sd_difference", "ci95_low", "ci95_high", "seeds_a_lower"]]),
        "",
        "## Pairing checks",
        _md(pd.DataFrame(ctx["pairing"]).T.reset_index().rename(columns={"index": "run_seed"})) if ctx["pairing"]
        else "no paired seeds",
        "",
        f"## Runtime\n{_md(ctx['runtime'], 1)}",
        "",
    ]
    if ctx["not_complete"]:
        lines += ["## Failed or incomplete runs (excluded)",
                  *[f"- `{d}` ({reason})" for d, _, reason in ctx["not_complete"]], ""]
    if ctx["unpaired"]:
        lines += [f"## Unpaired seeds (excluded): {ctx['unpaired']}", ""]
    lines += [
        "## Notes and limitations",
        "- Negative predictions are kept unclamped in all metrics; their frequency is in the seed table "
        "(`n_negative_predictions`).",
        f"- Daylight proxy: {dm['daylight_proxy']}.",
        f"- Large ramps: |y(t+1) - y(t)| >= {_fmt(dm['ramp_threshold_mwh'])} MWh ({dm['ramp_threshold_method']}).",
        "- Seed variability measures training randomness on one fixed period, not generalization across years.",
        "- Weather is a five-point spatial proxy, not plant locations or capacity weights; IESO totals cover "
        "registered generators >= 20 MW only, and telemetry-flagged hours are excluded.",
        "- Alpha (write proportion) plots are exploratory and do not show causal memory improvement.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------------------------

def analyze(runs_dir, out_dir=None):
    runs_dir = Path(runs_dir)
    runs, not_complete = discover_runs(runs_dir)
    if not runs:
        raise SystemExit(f"no verified complete solar-generation runs under {runs_dir}")
    missing = [m for m in MODELS if not any(model == m for _, model in runs)]
    if missing:
        raise SystemExit(f"no completed runs for {missing}; run each model with run_sweep.py (same seeds) first")
    out_dir = Path(out_dir) if out_dir else runs_dir / "analysis"
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    seeds = sorted({s for s, _ in runs})
    pairing, unpaired = {}, []
    for seed in seeds:
        if (seed, "qlstm") in runs and (seed, "qslstm") in runs:
            pairing[seed] = verify_pairing(runs[(seed, "qlstm")][0], runs[(seed, "qslstm")][0])
        else:
            unpaired.append(seed)
    (out_dir / "pairing_checks.json").write_text(json.dumps(pairing, indent=2), encoding="utf-8")
    accepted = [s for s, c in pairing.items() if c["accepted"]]
    rejected = [s for s, c in pairing.items() if not c["accepted"]]
    if rejected:
        print(f"WARNING: pairs for seeds {rejected} failed pairing checks and are excluded", file=sys.stderr)
    if unpaired:
        print(f"WARNING: seeds {unpaired} lack a completed run for one model and are excluded", file=sys.stderr)
    keys = [(s, m) for s in accepted for m in MODELS]

    per_seed = load_table(runs, keys, "per_seed_metrics_test.csv")
    per_seed.to_csv(out_dir / "combined_seed_metrics.csv", index=False)
    paired_rows, primary_table, primary = [], None, None
    for subset, group_type, group, metric in COMPARISONS:
        table, summary = sm.paired_difference(per_seed, metric, subset, group_type, group)
        paired_rows.append(summary)
        if (subset, group_type, metric) == ("all_eligible", "overall", "mse"):
            primary_table, primary = table, summary
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(out_dir / "paired_comparison.csv", index=False)
    primary_table.to_csv(out_dir / "primary_paired_mse.csv", index=False)

    overall = _overall(per_seed)
    model_summary = (overall.groupby("model")[["n", "mse", "mae", "rmse", "r2", "n_negative_predictions"]]
                     .agg(["mean", "std"]))
    model_summary.columns = [f"{a}_{b}".replace("_std", "_sd") for a, b in model_summary.columns]
    model_summary = model_summary.reset_index()
    model_summary.insert(1, "n_seeds", overall.groupby("model")["run_seed"].nunique().to_numpy())
    model_summary.to_csv(out_dir / "model_summary.csv", index=False)
    common = per_seed[(per_seed["subset"] == "baseline_common") & (per_seed["group_type"] == "overall")]
    baseline_table = (common.assign(forecaster=np.where(common["forecaster"] == "model", common["model"],
                                                        common["forecaster"]))
                      .groupby("forecaster")[["n", "mse", "mae", "rmse"]].mean().reset_index())

    preds = load_table(runs, keys, "predictions_test.csv")
    histories = []
    for k in keys:
        h = pd.read_csv(runs[k][0] / "history.csv")
        histories.append(h.assign(run_seed=k[0], model=k[1]))
    histories = pd.concat(histories, ignore_index=True) if histories else pd.DataFrame()
    alpha = load_table(runs, keys, "alpha_metrics_test.csv")
    dataset_manifest = json.loads((runs[keys[0]][0] / "dataset_manifest.json").read_text(encoding="utf-8")) \
        if keys else json.loads((next(iter(runs.values()))[0] / "dataset_manifest.json").read_text(encoding="utf-8"))

    if keys:
        plot_seed_mse(per_seed, out_dir / "plots" / "seed_mse.png")
        plot_paired_differences(primary_table, out_dir / "plots" / "paired_mse_differences.png")
        plot_learning_curves(histories, out_dir / "plots" / "learning_curves.png")
        plot_group_error(per_seed, out_dir / "plots" / "monthly_daylight_error.png")
        block = dataset_manifest["trace_block"]
        if block["n_hours"]:
            plot_trace(preds, block, min(accepted), out_dir / "plots" / "trace_test.png")
        if not alpha.empty:
            plot_alpha(alpha, out_dir / "plots" / "alpha_by_input_step.png")
        print(f"trace block ({'complete 7-day' if block['complete_7_day_block'] else 'fallback'}): "
              f"{block.get('first_target_timestamp_utc')} .. {block.get('last_target_timestamp_utc')}")

    runtime = []
    for (seed, model), (run_dir, _) in sorted(runs.items()):
        t = json.loads((run_dir / "timing.json").read_text(encoding="utf-8"))
        runtime.append({"run_seed": seed, "model": model, "train_seconds": t["train_seconds"],
                        "evaluation_seconds": t["evaluation_seconds"], "total_seconds": t["total_seconds"]})
    runtime = pd.DataFrame(runtime)
    runtime.to_csv(out_dir / "runtime.csv", index=False)
    revisions = {str(json.loads((d / "source_revision.json").read_text(encoding="utf-8")).get("commit"))
                 for d, _ in runs.values()}
    dirty = any(json.loads((d / "source_revision.json").read_text(encoding="utf-8")).get("dirty") for d, _ in runs.values())
    first_cfg = next(iter(runs.values()))[1]
    write_summary(out_dir / "summary.md", {
        "study": first_cfg["study"], "study_id": first_cfg["study_id"], "dataset_manifest": dataset_manifest,
        "primary_summary": primary, "primary_table": primary_table, "model_summary": model_summary,
        "baseline_table": baseline_table, "paired": paired, "pairing": pairing, "runtime": runtime,
        "not_complete": not_complete, "unpaired": unpaired, "requested": requested_seeds(runs_dir),
        "revision": f"commit(s) {', '.join(sorted(revisions))}; dirty working tree: {dirty}",
    })
    print(f"analysis written to {out_dir}")
    return out_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description="Analyze one solar-generation study.")
    parser.add_argument("--runs-dir", required=True, help="the study directory printed by run_sweep.py")
    parser.add_argument("--out-dir", default=None, help="default: <runs-dir>/analysis")
    args = parser.parse_args(argv)
    analyze(args.runs_dir, args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
