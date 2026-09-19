# src/q_slstm/trainers/train.py

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from typing import Union

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .utils import log_epoch, plot_losses, predict_and_log


@dataclass
class LoaderBundle:
	train_loader: DataLoader
	test_loader: DataLoader
	simulation_loader: DataLoader
	train_len: int


def make_loaders(args, dataset_bundle) -> LoaderBundle:

	train_loader = DataLoader(
		dataset_bundle.train_ds,
		batch_size=args.batch_size,
		shuffle=True,
		num_workers=getattr(args, "num_workers", 0)
	)
	test_loader = DataLoader(
		dataset_bundle.test_ds,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=getattr(args, "num_workers", 0)
	)
	simulation_loader = DataLoader(
		dataset_bundle.simulation_ds,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=getattr(args, "num_workers", 0)
	)

	return LoaderBundle(
		train_loader=train_loader,
		test_loader=test_loader,
		simulation_loader=simulation_loader,
		train_len=dataset_bundle.train_len,
	)


def _extract_model_output(args, out):
	"""
	Normalize model forward outputs into a tensor aligned with y.
	This keeps your old per-model handling in one place.
	"""
	if args.model in ("qlstm", "qslstm", "lstm"):
		out, _ = out
		out = out[:, -1, :]
		return out
	elif args.model in (
		"self_modulating_qfwp",
		"self_modulating_qfwp_only_new_params",
		"self_modulating_qfwp_only_old_params",
		"standard_qfwp",
	):
		return out[-1]
	return out


def _align_prediction_and_target(out: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
	"""
	Keep output/target shapes compatible for both single-target and multi-target training.
	"""
	out = out.float()
	y = y.float()

	if out.ndim == 1:
		out = out.unsqueeze(-1)
	if y.ndim == 1:
		y = y.unsqueeze(-1)

	return out, y


def run_training(
	args,
	model: torch.nn.Module,
	loaders: LoaderBundle,
	result_path: Union[str, os.PathLike],
	logger,
) -> None:
	# os.makedirs(result_path, exist_ok=True)
	result_path.mkdir(parents=True, exist_ok=True)

	# save args snapshot
	logger.info("Args: %s", json.dumps(vars(args), indent=4))

	model = model.to(args.device)

	criterion = nn.MSELoss()
	optimizer = optim.Adam(model.parameters(), lr=args.lr)

	train_losses = []
	test_losses = []

	# csv_path = os.path.join(result_path, "train_log.csv")
	# prediction_csv_path = os.path.join(result_path, "prediction_log.csv")
	csv_path = result_path / "train_log.csv"
	prediction_csv_path = result_path / "prediction_log.csv"

	for epoch in range(1, args.epochs + 1):
		# ---- train ----
		model.train()
		train_loss = 0.0

		for X, y in loaders.train_loader:
			X = X.to(args.device, non_blocking=True)
			y = y.to(args.device, non_blocking=True)

			optimizer.zero_grad()
			out = model(X)
			out = _extract_model_output(args, out)
			out, y = _align_prediction_and_target(out, y)

			loss = criterion(out, y)
			loss.backward()
			optimizer.step()

			train_loss += loss.item() * X.size(0)

		train_loss /= len(loaders.train_loader.dataset)

		# ---- eval ----
		model.eval()
		test_loss = 0.0
		with torch.no_grad():
			for X, y in loaders.test_loader:
				X = X.to(args.device, non_blocking=True)
				y = y.to(args.device, non_blocking=True)

				out = model(X)
				out = _extract_model_output(args, out)
				out, y = _align_prediction_and_target(out, y)

				loss = criterion(out, y)
				test_loss += loss.item() * X.size(0)

		test_loss /= len(loaders.test_loader.dataset)

		print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

		logger.info("Epoch %d: train loss=%.6f, test loss=%.6f", epoch, train_loss, test_loss)
		log_epoch(epoch, train_loss, test_loss, csv_path)

		train_losses.append(train_loss)
		test_losses.append(test_loss)

		# prediction plot/log
		# prediction_plot_path = os.path.join(result_path, f"prediction_plot_epoch_{epoch}.png")
		prediction_plot_path = result_path / f"prediction_plot_epoch_{epoch}.png"
		predict_and_log(
			args=args,
			model=model,
			loader=loaders.simulation_loader,
			train_len=loaders.train_len,
			csv_path=prediction_csv_path,
			split="simulation",
			epoch=epoch,
			debug_path=prediction_plot_path,
		)

		# loss plot
		# loss_plot_path = os.path.join(result_path, f"loss_compare_plot_epoch_{epoch}.png")
		loss_plot_path = result_path / f"loss_compare_plot_epoch_{epoch}.png"
		plot_losses(train_losses, test_losses, epoch, save_path=loss_plot_path)

		# checkpoint
		# model_path = os.path.join(result_path, f"saved_checkpoint_epoch_{epoch}.pth")
		model_path = result_path / f"saved_checkpoint_epoch_{epoch}.pth"
		torch.save(
			{
				"epoch": epoch,
				"model_state_dict": model.state_dict(),
				"optimizer_state_dict": optimizer.state_dict(),
				"loss": float(loss.detach().cpu().item()),
			},
			model_path,
		)

