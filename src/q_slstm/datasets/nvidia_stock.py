from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset


class NVidiaStockSequenceDataset(Dataset):
	"""
	Builds a multivariate sequence dataset from NVidia stock CSV.
	Uses all numeric stock features as input and next-step Adj Close as target.
	"""

	def __init__(
		self,
		csv_path,
		seq_len: int = 4,
		target_col: str = "Adj Close",
		feature_range=(-1, 1),
		dtype=torch.float32,
	):
		df = pd.read_csv(csv_path)
		if "Date" in df.columns:
			df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
			df = df.sort_values("Date")

		feature_cols = ["Adj Close", "Close", "High", "Low", "Open", "Volume"]
		missing_cols = [c for c in feature_cols if c not in df.columns]
		if missing_cols:
			raise ValueError(f"Missing feature columns: {missing_cols}")
		if target_col not in df.columns:
			raise ValueError(f"Missing target column: {target_col}")

		features = df[feature_cols].astype(float).to_numpy()
		target = df[target_col].astype(float).to_numpy().reshape(-1)

		self.x_scaler = MinMaxScaler(feature_range=feature_range)
		self.y_scaler = MinMaxScaler(feature_range=feature_range)
		features_scaled = self.x_scaler.fit_transform(features)
		target_scaled = self.y_scaler.fit_transform(target.reshape(-1, 1)).reshape(-1)

		self.seq_len = int(seq_len)
		self.dtype = dtype

		xs, ys = [], []
		for i in range(len(features_scaled) - self.seq_len - 1):
			xs.append(features_scaled[i : i + self.seq_len])
			ys.append(target_scaled[i + self.seq_len])

		self.x = torch.tensor(np.array(xs), dtype=self.dtype)
		self.y = torch.tensor(np.array(ys), dtype=self.dtype)

	def __len__(self):
		return self.x.shape[0]

	def __getitem__(self, idx):
		return self.x[idx], self.y[idx]
