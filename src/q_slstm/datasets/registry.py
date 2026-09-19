# src/q_slstm/datasets/registry.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
from typing import Union

import torch
from torch.utils.data import ConcatDataset

# TODO: 把你原本 data.* 的 import 改成 q_slstm.datasets.* 或留著等你搬完再改
# Sri Lanka Data
from q_slstm.datasets.sri_dataset import prepare_sequence_region
from q_slstm.datasets.sri_dataset import list_regions
from q_slstm.datasets.sri_dataset import prepare_tabular_region

# Bessel / SHM / DQC / NARMA
from q_slstm.datasets.bessel_functions import BesselSequenceDataset
from q_slstm.datasets.damped_shm import DampedSHMSequenceDataset
from q_slstm.datasets.delayed_quantum_control import DelayedQCDataset
from q_slstm.datasets.narma_generator import make_narma_dataset
from q_slstm.datasets.nvidia_stock import NVidiaStockSequenceDataset
from q_slstm.datasets.weather_station import WeatherStationSequenceDataset

from q_slstm.config import DATA # 2026 02 22: Auto detect the path.

@dataclass
class DatasetBundle:
	train_ds: torch.utils.data.Dataset
	test_ds: torch.utils.data.Dataset
	simulation_ds: torch.utils.data.Dataset
	train_len: int


def make_datasets(args) -> DatasetBundle:
	"""
	Create datasets based on args.dataset.
	Returns DatasetBundle(train_ds, test_ds, simulation_ds, train_len).
	"""
	train_ds = None
	test_ds = None
	simulation_ds = None
	train_len = 0

	if args.dataset == "sri_lanka":
		# TODO: 搬運你原本 Sri Lanka 的 prepare_sequence_region 邏輯
		# seq_r = prepare_sequence_region(...)
		# train_ds = seq_r["train"]
		# test_ds = seq_r["val"]
		# train_len = len(train_ds)
		# simulation_ds = ConcatDataset([train_ds, test_ds])
		CSV_PATH = DATA / "sri_lanka_2013-2022_vertical.csv"
		seq_r = prepare_sequence_region(
			CSV_PATH,
			region_name=args.region,
			feature_cols=[
				'meanTair_F_Inst','minTair_F_Inst','maxTair_F_Inst',
				'meanQair_F_Inst','meanSoilmoi0_10Cm_Inst','meanCanopint_Inst',
				'meanPsurf_F_Inst',
				'meanNdvi','minNdvi','maxNdvi',
				'meanPrecipitationcal','minPrecipitationcal','maxPrecipitationcal'
				],
			window_len=args.window_len,    # 想要 2022 也有 test 序列時，8 會更穩妥
			horizon=args.horizon,
			train_end="2020-12-31",
			val_end="2021-12-31",
			y_scaler_type = "minmax")

		train_ds = seq_r["train"]
		test_ds = seq_r["val"]
		train_len = len(train_ds)
		simulation_ds = ConcatDataset([train_ds, test_ds])
		# raise NotImplementedError("TODO: migrate sri_lanka dataset pipeline")

	elif args.dataset == "bessel_j2":
		j2_dataset = BesselSequenceDataset(seq_len = args.window_len)

		n_total = len(j2_dataset)
		n_train = int(0.8 * n_total)
		train_len = n_train

		train_ds = torch.utils.data.Subset(j2_dataset, range(0, n_train))
		test_ds  = torch.utils.data.Subset(j2_dataset, range(n_train, n_total))
		simulation_ds = j2_dataset
		# raise NotImplementedError("TODO: migrate bessel_j2 dataset pipeline")

	elif args.dataset == "damped_shm":
		damped_shm_dataset = DampedSHMSequenceDataset(seq_len = args.window_len)

		n_total = len(damped_shm_dataset)
		n_train = int(0.8 * n_total)
		train_len = n_train

		train_ds = torch.utils.data.Subset(damped_shm_dataset, range(0, n_train))
		test_ds  = torch.utils.data.Subset(damped_shm_dataset, range(n_train, n_total))
		simulation_ds = damped_shm_dataset
		# raise NotImplementedError("TODO: migrate damped_shm dataset pipeline")

	elif args.dataset == "delayed_quantum_control":
		delayed_quantum_control_dataset = DelayedQCDataset(seq_len = args.window_len)

		n_total = len(delayed_quantum_control_dataset)
		n_train = int(0.8 * n_total)
		train_len = n_train

		train_ds = torch.utils.data.Subset(delayed_quantum_control_dataset, range(0, n_train))
		test_ds  = torch.utils.data.Subset(delayed_quantum_control_dataset, range(n_train, n_total))
		simulation_ds = delayed_quantum_control_dataset
		# raise NotImplementedError("TODO: migrate delayed_quantum_control dataset pipeline")

	elif args.dataset == "narma_5":
		narma_5_dataset, _ = make_narma_dataset(n_0 = 5, seq_len = args.window_len)

		n_total = len(narma_5_dataset)
		n_train = int(0.8 * n_total)
		train_len = n_train

		train_ds = torch.utils.data.Subset(narma_5_dataset, range(0, n_train))
		test_ds  = torch.utils.data.Subset(narma_5_dataset, range(n_train, n_total))
		simulation_ds = narma_5_dataset
		# raise NotImplementedError("TODO: migrate narma_5 dataset pipeline")

	elif args.dataset == "narma_10":
		narma_10_dataset, _ = make_narma_dataset(n_0 = 10, seq_len = args.window_len)

		n_total = len(narma_10_dataset)
		n_train = int(0.8 * n_total)
		train_len = n_train

		train_ds = torch.utils.data.Subset(narma_10_dataset, range(0, n_train))
		test_ds  = torch.utils.data.Subset(narma_10_dataset, range(n_train, n_total))
		simulation_ds = narma_10_dataset
		# raise NotImplementedError("TODO: migrate narma_10 dataset pipeline")

	elif args.dataset == "nvidia_stock":
		nvidia_stock_dataset = NVidiaStockSequenceDataset(
			csv_path=DATA / "NVidia_stock_history.csv",
			seq_len=args.window_len,
			target_col="Adj Close",
		)

		n_total = len(nvidia_stock_dataset)
		n_train = int(0.8 * n_total)
		train_len = n_train

		train_ds = torch.utils.data.Subset(nvidia_stock_dataset, range(0, n_train))
		test_ds = torch.utils.data.Subset(nvidia_stock_dataset, range(n_train, n_total))
		simulation_ds = nvidia_stock_dataset

	elif args.dataset == "weather_station":
		weather_dataset = WeatherStationSequenceDataset(
			data_dir=args.weather_data_dir,
			seq_len=args.window_len,
			horizon=args.horizon,
		)

		n_total = len(weather_dataset)
		n_train = int(0.8 * n_total)
		train_len = n_train

		train_ds = torch.utils.data.Subset(weather_dataset, range(0, n_train))
		test_ds = torch.utils.data.Subset(weather_dataset, range(n_train, n_total))
		simulation_ds = weather_dataset

	else:
		raise ValueError(f"Unknown dataset: {args.dataset}")

	return DatasetBundle(
		train_ds=train_ds,
		test_ds=test_ds,
		simulation_ds=simulation_ds,
		train_len=train_len,
	)
