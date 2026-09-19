# scripts/experiments/time_series/train_timeseries.py

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn

from q_slstm.datasets.registry import make_datasets
from q_slstm.trainers.train import make_loaders, run_training
from q_slstm.trainers.utils import setup_logger

import json
from contextlib import redirect_stdout, redirect_stderr

from q_slstm.utils.experiment import save_args_json
from q_slstm.utils.experiment import save_git_revision
from q_slstm.utils.experiment import save_environment_snapshot
from q_slstm.utils.experiment import build_result_path
from q_slstm.utils.experiment import generate_experiment_readme
from q_slstm.utils.experiment import Tee

from q_slstm.models.qlstm_cell import CustomQLSTMCell
from q_slstm.models.q_slstm_cell import CustomQsLSTMCell, DEFAULT_GATE_EPSILON
from q_slstm.models.sequence_wrappers import CustomLSTM, CustomQsLSTM

class StandardLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        self.output_post_processing = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        h_t, c_t = self.lstm_cell(x, hidden)
        out = self.output_post_processing(h_t)
        return out, h_t, c_t


class InputProjectionWrapper(nn.Module):
    """
    Learns a projection from high-dimensional features to a smaller input for the recurrent model.
    """

    def __init__(self, input_size: int, projected_input_size: int, base_model: nn.Module):
        super().__init__()
        self.input_projection = nn.Linear(input_size, projected_input_size)
        self.base_model = base_model

    def forward(self, x, hidden=None):
        projected_x = self.input_projection(x)
        return self.base_model(projected_x, hidden)


def make_model(args):
    """
    Create model based on args.model.
    Move your existing model-creation if/elif here.
    """
    use_input_projection = (
        args.model in ("qlstm", "qslstm", "lstm")
        and args.input_projection_size > 0
        and args.input_size > args.input_projection_size
    )
    effective_input_size = args.input_projection_size if use_input_projection else args.input_size
    if use_input_projection:
        print(
            f"Applying input projection for {args.model}: "
            f"{args.input_size} -> {effective_input_size}"
        )

    if args.model == "qlstm":
        # Conventional QLSTM baseline: sigmoid gates, (h, c) state.
        print("OPERATING MODEL {}".format(args.model))
        qlstm_cell = CustomQLSTMCell(
            effective_input_size, args.hidden_size, args.output_size, args.qnn_depth
        ).float()
        base_model = CustomLSTM(effective_input_size, args.hidden_size, qlstm_cell).float()
        model = (
            InputProjectionWrapper(args.input_size, effective_input_size, base_model)
            if use_input_projection
            else base_model
        )
        return model.to(args.device).float()
    elif args.model == "qslstm":
        # Q-sLSTM: bounded log gates, xLSTM stabilizer, (h, c, n, m) state.
        print("OPERATING MODEL {}".format(args.model))
        qslstm_cell = CustomQsLSTMCell(
            effective_input_size,
            args.hidden_size,
            args.output_size,
            args.qnn_depth,
            gate_epsilon=getattr(args, "gate_epsilon", DEFAULT_GATE_EPSILON),
        ).float()
        base_model = CustomQsLSTM(effective_input_size, args.hidden_size, qslstm_cell).float()
        model = (
            InputProjectionWrapper(args.input_size, effective_input_size, base_model)
            if use_input_projection
            else base_model
        )
        return model.to(args.device).float()
    elif args.model == "lstm":
        print("OPERATING MODEL {}".format(args.model))
        lstm_cell = StandardLSTMCell(effective_input_size, args.hidden_size, args.output_size).float()
        base_model = CustomLSTM(effective_input_size, args.hidden_size, lstm_cell).float()
        model = (
            InputProjectionWrapper(args.input_size, effective_input_size, base_model)
            if use_input_projection
            else base_model
        )
        return model.to(args.device).float()
    elif args.model == "self_modulating_qfwp":
        raise NotImplementedError("TODO: migrate SelfModulatingFWP model creation")
    elif args.model == "self_modulating_qfwp_only_old_params":
        raise NotImplementedError("TODO: migrate SelfModulatingFWPOnlyOldParameters model creation")
    elif args.model == "self_modulating_qfwp_only_new_params":
        raise NotImplementedError("TODO: migrate SelfModulatingFWPOnlyNewParameters model creation")
    elif args.model == "standard_qfwp":
        raise NotImplementedError("TODO: migrate standard_qfwp model creation")


    else:
        raise ValueError(f"Unknown model: {args.model}")


def main():
    parser = argparse.ArgumentParser(description="Time-series training entrypoint.")

    parser.add_argument("--epochs", type=int, default=100, help="訓練的迭代次數 (default: 10)")
    parser.add_argument("--lr", type=float, default=1e-3, help="學習率 (default: 0.001)")

    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "qlstm",
            "qslstm",
            "lstm",
            "self_modulating_qfwp",
            "self_modulating_qfwp_only_new_params",
            "self_modulating_qfwp_only_old_params",
            "standard_qfwp",
        ],
        default="lstm",
        help="選擇模型類別 (default: qlstm)"
    )

    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cpu")

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["bessel_j2", "damped_shm", "delayed_quantum_control", "narma_5", "narma_10", "sri_lanka", "nvidia_stock", "weather_station"],
        default="bessel_j2",
        help="Choose a dataset (default: bessel_j2)"
    )

    parser.add_argument(
        "--weather_data_dir",
        type=str,
        default="weather_data",
        help="Path to organized weather station CSV files for weather_station dataset"
    )

    parser.add_argument("--save_dir", type=str, default="results", help="實驗輸出存放的資料夾")
    parser.add_argument("--exp_name", type=str, default="experiment", help="Name of the experiment")

    parser.add_argument("--region", type=str, default="Colombo", help="（資料）地區選擇 (Only for Sri Lanka dataset)")
    parser.add_argument("--window_len", type=int, default=4, help="Length of sliding window")
    parser.add_argument("--horizon", type=int, default=1, help="Predicting horizon")

    parser.add_argument("--input_size", type=int, default=1, help="Input size (dimension)")
    parser.add_argument("--hidden_size", type=int, default=5, help="Hidden size (QLSTM) also for QFWP")
    parser.add_argument("--output_size", type=int, default=1, help="Output size (QLSTM) also for QFWP")
    parser.add_argument("--qnn_depth", type=int, default=5, help="QNN depth (in QLSTM) also for QFWP")
    parser.add_argument(
        "--input_projection_size",
        type=int,
        default=6,
        help="If input_size is larger than this, qlstm project inputs before quantum cell",
    )

    parser.add_argument(
        "--gate_epsilon",
        type=float,
        default=DEFAULT_GATE_EPSILON,
        help="Epsilon in (0, 1) bounding the quantum input/forget log-ratio transforms and the normalizer division (qslstm only, default: 1e-6)",
    )

    parser.add_argument("--batch_size", type=int, default=2, help="Batch size (large batch size may also need larger learning rate)")
    parser.add_argument("--seed", type=int, default=42, help="random seed (default:42)")
    parser.add_argument("--debug", action="store_true", help="開啟除錯模式")


    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of worker processes for DataLoader (default: 0)"
    )

    args = parser.parse_args()

    # seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # output folder
    EXPERIMENT_ROOT = Path(__file__).resolve().parent
    result_path = build_result_path(args, EXPERIMENT_ROOT)

    # 儲存實驗設定
    save_args_json(args, result_path)
    save_git_revision(result_path)
    save_environment_snapshot(result_path)
    generate_experiment_readme(args, result_path)

    # Console log
    console_log_path = result_path / "console_log.txt"
    log_f = open(console_log_path, "a", buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_f)
    sys.stderr = Tee(sys.__stderr__, log_f)

    # logger
    log_path = result_path / "log.txt"
    logger = setup_logger(log_path)

    # datasets + loaders
    dataset_bundle = make_datasets(args)
    loaders = make_loaders(args, dataset_bundle)

    # auto-set input size for dataset-specific feature dimensions
    if hasattr(dataset_bundle.train_ds, "dataset"):
        dataset = dataset_bundle.train_ds.dataset
    else:
        dataset = dataset_bundle.train_ds
    if hasattr(dataset, "input_size"):
        args.input_size = dataset.input_size
    if hasattr(dataset, "output_size"):
        args.output_size = dataset.output_size
    if hasattr(dataset, "feature_cols") and hasattr(dataset, "target_cols"):
        print(f"Feature count: {len(dataset.feature_cols)}")
        print(f"Target count: {len(dataset.target_cols)}")
        print(f"Targets: {dataset.target_cols}")

    # Refresh metadata after dataset auto-detection so args.json records
    # the real feature/target dimensions used by training.
    save_args_json(args, result_path)
    generate_experiment_readme(args, result_path)

    # model
    model = make_model(args)

    # train
    run_training(args=args, model=model, loaders=loaders, result_path=result_path, logger=logger)


if __name__ == "__main__":
    main()
