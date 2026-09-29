# scripts/experiments/nearest_neighbor/classical_baseline.py
#
# Classical nn.LSTM baseline on exactly the data of existing quantum runs (datasets rebuilt from each run's
# config.json), with the same masked MSE, Adam, batch size, epoch budget and best-validation checkpoint.
# Sweeps hidden size x learning rate, and reports validation / test MSE per case type next to the
# constant predictor (mean training target), so every number can be read as "how much better than nothing".

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import torch.nn.functional as F  # noqa: E402

from q_slstm.experiments.nearest_neighbor import make_datasets, masked_mse  # noqa: E402
from q_slstm.models.factory import build_quantum_model  # noqa: E402
from q_slstm.models.q_slstm_cell import stabilize_gates  # noqa: E402

ARCHS = ("lstm", "slstm_sig", "slstm_exp")
N_FLOOR = 1e-6


class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.head(h), None


class SLSTMRegressor(nn.Module):
    """Classical xLSTM sLSTM: exponential input gate, normalizer n, log-domain stabilizer m, h = o * c / n.

    The same recurrence as the quantum Q-sLSTM with linear layers in place of VQCs. `exp_forget` selects
    f = exp(f~) (xLSTM default; analogue of the amplified-forget runs) over f = sigmoid(f~).
    """

    def __init__(self, input_size, hidden_size, output_size, exp_forget):
        super().__init__()
        self.hidden_size = hidden_size
        self.exp_forget = exp_forget
        self.gates = nn.Linear(input_size + hidden_size, 4 * hidden_size)
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch, length, _ = x.shape
        h = c = n = m = x.new_zeros(batch, self.hidden_size)
        outputs = []
        for t in range(length):
            pre_i, pre_f, pre_z, pre_o = self.gates(torch.cat((x[:, t], h), dim=-1)).chunk(4, dim=-1)
            ell_f = pre_f if self.exp_forget else F.logsigmoid(pre_f)
            m, i_prime, f_prime = stabilize_gates(pre_i, ell_f, m)
            c = f_prime * c + i_prime * torch.tanh(pre_z)
            n = f_prime * n + i_prime
            h = torch.sigmoid(pre_o) * (c / torch.clamp_min(n, N_FLOOR))
            outputs.append(h)
        return self.head(torch.stack(outputs, dim=1)), None


def build(arch, input_size, hidden_size, output_size):
    if arch == "lstm":
        return LSTMRegressor(input_size, hidden_size, output_size)
    return SLSTMRegressor(input_size, hidden_size, output_size, exp_forget=arch == "slstm_exp")


def masked(tensor_ds, key):
    return tensor_ds.tensors[key][..., 0].numpy().astype(bool)


def evaluate(model, ds):
    model.eval()
    with torch.no_grad():
        pred = model(ds.tensors["inputs"])[0][..., 0].numpy()
    y, mask = ds.tensors["targets"][..., 0].numpy(), masked(ds, "metric_mask")
    se = (pred - y) ** 2
    out = {"all": float(se[mask].mean())}
    cases = np.array(ds.case_types)
    for case in np.unique(cases):
        sel = cases == case
        out[case] = float(se[sel][mask[sel]].mean())
    return out


def constant_baseline(datasets):
    train = datasets["train"]
    c = float(train.tensors["targets"][..., 0].numpy()[masked(train, "metric_mask")].mean())
    out = {}
    for split in ("val", "test"):
        ds = datasets[split]
        y, mask = ds.tensors["targets"][..., 0].numpy(), masked(ds, "metric_mask")
        se = (y - c) ** 2
        out[split] = {"all": float(se[mask].mean())}
        cases = np.array(ds.case_types)
        for case in np.unique(cases):
            sel = cases == case
            out[split][case] = float(se[sel][mask[sel]].mean())
    return out


def train_one(datasets, config, arch, hidden, lr, epochs, seed):
    torch.manual_seed(seed)
    model = build(arch, config["input_size"], hidden, config["output_size"])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(datasets["train"], batch_size=config["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    best, best_state, best_epoch, curve = np.inf, None, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for batch in loader:
            optimizer.zero_grad()
            loss = masked_mse(model(batch["inputs"])[0], batch["targets"], batch["loss_mask"])
            loss.backward()
            optimizer.step()
            total += loss.item() * batch["inputs"].shape[0]
            n += batch["inputs"].shape[0]
        val = evaluate(model, datasets["val"])["all"]
        curve.append({"epoch": epoch, "train_loss": total / n, "val_mse": val})
        if val < best:
            best, best_state, best_epoch = val, copy.deepcopy(model.state_dict()), epoch
    model.load_state_dict(best_state)
    n_params = sum(p.numel() for p in model.parameters())
    return model, best_epoch, n_params, pd.DataFrame(curve)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", nargs="+", type=Path, required=True,
                        help="existing quantum run directories whose config.json defines the data")
    parser.add_argument("--arch", nargs="+", choices=ARCHS, default=["lstm"])
    parser.add_argument("--hidden", nargs="+", type=int, default=[6, 32, 128])
    parser.add_argument("--lr", nargs="+", type=float, default=[1e-2, 1e-3])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)

    rows, curves = [], []
    for run_dir in args.run_dirs:
        config = json.loads((run_dir / "config.json").read_text())
        datasets = make_datasets({**config, "run_extrapolation": False})
        seed = config["seeds"]["run_seed"]
        quantum = build_quantum_model(config["model"], config["input_size"], config["hidden_size"],
                                      config["output_size"], config["qnn_depth"], gate_epsilon=config["gate_epsilon"])
        const = constant_baseline(datasets)
        rows.append({"model": "constant", "run_seed": seed, "val_mse": const["val"]["all"],
                     **{f"test_{k}": v for k, v in const["test"].items()}})
        print(f"seed {seed}: constant val {const['val']['all']:.4f} test {const['test']['all']:.4f}; "
              f"quantum {config['model']} params {sum(p.numel() for p in quantum.parameters())}", flush=True)
        for arch, hidden, lr in ((a, h, r) for a in args.arch for h in args.hidden for r in args.lr):
                start = time.perf_counter()
                model, best_epoch, n_params, curve = train_one(datasets, config, arch, hidden, lr, args.epochs, seed)
                val, test = evaluate(model, datasets["val"]), evaluate(model, datasets["test"])
                rows.append({"model": f"{arch}_h{hidden}_lr{lr:g}", "run_seed": seed, "arch": arch, "hidden": hidden, "lr": lr,
                             "n_params": n_params, "best_epoch": best_epoch, "val_mse": val["all"],
                             **{f"test_{k}": v for k, v in test.items()}})
                curves.append(curve.assign(run_seed=seed, arch=arch, hidden=hidden, lr=lr))
                print(f"  {arch:<9} h={hidden:<4} lr={lr:<6g} params={n_params:<6} best@{best_epoch:<3} val {val['all']:.4f} "
                      f"test {test['all']:.4f}  ({time.perf_counter() - start:.0f}s)", flush=True)
                pd.DataFrame(rows).to_csv(args.out_dir / "classical_baseline_per_seed.csv", index=False)
                pd.concat(curves).to_csv(args.out_dir / "classical_baseline_curves.csv", index=False)

    table = pd.DataFrame(rows)
    summary = table.groupby("model", sort=False).mean(numeric_only=True).drop(columns="run_seed")
    summary.to_csv(args.out_dir / "classical_baseline_summary.csv")
    with pd.option_context("display.width", 250):
        print(summary.round(4).to_string())
    print(f"written to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
