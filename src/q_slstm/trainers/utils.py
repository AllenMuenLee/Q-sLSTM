from __future__ import annotations

import csv
import logging
import os
from typing import List, Optional

from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import torch

def log_epoch(epoch, train_loss, test_loss, path):
	import os, csv

	# file_exists = os.path.isfile(path)
	file_exists = path.is_file()
	with open(path, "a", newline="") as f:
		writer = csv.writer(f)
		# 如果是新檔案就加上表頭
		if not file_exists:
			writer.writerow(["epoch", "train_loss", "test_loss"])
		writer.writerow([epoch, train_loss, test_loss])

def setup_logger(path: str) -> logging.Logger:
	logger = logging.getLogger("train_logger")
	logger.setLevel(logging.INFO)
	logger.handlers.clear()

	fh = logging.FileHandler(path, mode="w")
	fh.setLevel(logging.INFO)
	fmt = logging.Formatter("%(asctime)s - %(message)s")
	fh.setFormatter(fmt)

	logger.addHandler(fh)
	logger.propagate = False
	return logger


def plot_losses(train_losses, test_losses, epoch, save_path):
	"""
	繪製並儲存 train/test loss 曲線
	"""
	plt.figure()
	plt.plot(range(1, len(train_losses)+1), train_losses, label="Train Loss")
	plt.plot(range(1, len(test_losses)+1), test_losses, label="Test Loss")
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.title("Training vs Test Loss")
	plt.legend()
	plt.savefig(save_path)
	plt.close()


def predict_and_log(
	args,
	model: torch.nn.Module,
	loader: torch.utils.data.DataLoader,
	train_len: int,
	csv_path: Union[str, os.PathLike],
	split: str = "simulation",
	epoch: Optional[int] = None,
	debug_plot: bool = True,
	debug_path: Union[str, os.PathLike] = "debug_plot.png",
) -> None:
	model.eval()
	all_ytrue, all_ypred = [], []
	base_dataset = getattr(loader.dataset, "dataset", loader.dataset)
	target_names = list(getattr(base_dataset, "target_cols", []))

	with torch.no_grad(), open(csv_path, "a", newline="") as f:
		writer = csv.writer(f)
		if f.tell() == 0:
			writer.writerow(
				["epoch", "model", "split", "horizon", "target_idx", "target_name", "y_true", "y_pred"]
			)

		for xb, yb in loader:
			xb = xb.to(args.device, non_blocking=True)
			yb = yb.to(args.device, non_blocking=True)

			yhat = model(xb)

			# NOTE: 這段就是你原本的「model-specific output handling」
			if args.model in ("qlstm", "qslstm", "qslstm_log", "lstm"):
				yhat, _ = yhat
				yhat = yhat[:, -1, :]
			elif args.model in (
				"self_modulating_qfwp",
				"self_modulating_qfwp_only_new_params",
				"self_modulating_qfwp_only_old_params",
				"standard_qfwp",
			):
				yhat = yhat[-1]

			yhat = yhat.detach().cpu()
			yb = yb.detach().cpu()

			if yhat.ndim == 1:
				yhat = yhat.unsqueeze(-1)
			if yb.ndim == 1:
				yb = yb.unsqueeze(-1)

			if yhat.shape != yb.shape:
				raise ValueError(f"Prediction/target shape mismatch: yhat={tuple(yhat.shape)}, y={tuple(yb.shape)}")

			for row_idx in range(yhat.shape[0]):
				for target_idx in range(yhat.shape[1]):
					target_name = (
						target_names[target_idx]
						if target_idx < len(target_names)
						else f"target_{target_idx}"
					)
					yt = float(yb[row_idx, target_idx].item())
					yp = float(yhat[row_idx, target_idx].item())
					writer.writerow(
						[epoch, args.model, split, args.horizon, target_idx, target_name, yt, yp]
					)
					if target_idx == 0:
						all_ytrue.append(yt)
						all_ypred.append(yp)

	if debug_plot and all_ytrue:
		plt.figure(figsize=(8, 4))
		plt.plot(all_ytrue, label="Ground Truth", linewidth=1.2)
		plt.plot(all_ypred, label="Prediction", linewidth=1.2)
		plt.axvline(x=train_len, c="r", linestyle="--")
		plt.title(f"[DEBUG] {args.model} {split} (epoch={epoch})")
		plt.xlabel("timestep (index)")
		plt.ylabel("value")
		plt.legend()
		plt.tight_layout()
		plt.savefig(debug_path, dpi=200)
		plt.close()
