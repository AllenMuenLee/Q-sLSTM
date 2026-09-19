from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset


class WeatherStationSequenceDataset(Dataset):
    """
    Multivariate weather dataset for next-day forecasting.
    Inputs are weather + calendar features; targets are next-day max/min temperature.
    """

    DEG = "\u00b0"
    RAW_COL_MAP = {
        "Date/Time": "date_iso",
        "Date/Time (LST)": "date_iso",
        f"Max Temp ({DEG}C)": "max_temp_c",
        f"Min Temp ({DEG}C)": "min_temp_c",
        f"Mean Temp ({DEG}C)": "mean_temp_c",
        f"Heat Deg Days ({DEG}C)": "heat_deg_days_c",
        f"Cool Deg Days ({DEG}C)": "cool_deg_days_c",
        "Total Rain (mm)": "total_rain_mm",
        "Total Snow (cm)": "total_snow_cm",
        "Total Precip (mm)": "total_precip_mm",
        "Snow on Grnd (cm)": "snow_on_grnd_cm",
        "Dir of Max Gust (10s deg)": "dir_max_gust_10deg",
        "Spd of Max Gust (km/h)": "spd_max_gust_kmh",
        "Latitude (y)": "latitude",
        "Longitude (x)": "longitude",
        "time_idx": "time_idx",
        "year_num": "year_num",
        "month_num": "month_num",
        "day_num": "day_num",
        "day_of_year": "day_of_year",
        "day_of_week": "day_of_week",
        "is_weekend": "is_weekend",
        "doy_sin": "doy_sin",
        "doy_cos": "doy_cos",
        "dow_sin": "dow_sin",
        "dow_cos": "dow_cos",
    }
    DATE_FEATURE_COLS = ["year_num", "doy_sin", "doy_cos"]

    TARGET_COLS = ["max_temp_c", "min_temp_c"]
    FEATURE_COLS = [
        "mean_temp_c",
        "max_temp_c",
        "min_temp_c",
        "heat_deg_days_c",
        "cool_deg_days_c",
        "doy_sin",
        "doy_cos",
        "year_num",
    ]

    def __init__(
        self,
        data_dir: str | Path | None = None,
        seq_len: int = 4,
        horizon: int = 1,
        feature_range=(-1, 1),
        dtype=torch.float32,
    ):
        self.seq_len = int(seq_len)
        self.horizon = int(horizon)
        self.dtype = dtype

        if data_dir is None:
            data_dir = Path(__file__).resolve().parents[3] / "weather_data"
        self.data_dir = Path(data_dir)

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Weather data directory not found: {self.data_dir}")

        csv_files = sorted(self.data_dir.glob("[0-9][0-9]_*.csv"))
        if not csv_files:
            csv_files = sorted(self.data_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in weather data directory: {self.data_dir}")

        required_cols = set(self.FEATURE_COLS + self.TARGET_COLS + ["date_iso"])
        selected_cols = ["date_iso"] + list(dict.fromkeys(self.FEATURE_COLS + self.TARGET_COLS))

        station_frames: list[pd.DataFrame] = []
        for csv_path in csv_files:
            df = pd.read_csv(csv_path, low_memory=False)
            # Some CSV exports include mojibake in degree symbols (e.g., "\u00C2\u00B0").
            df = df.rename(
                columns=lambda c: c.replace("\u00C2\u00B0", "\u00B0").strip() if isinstance(c, str) else c
            )
            if "Date/Time" in df.columns and "date_iso" in df.columns:
                # Prefer Date/Time from source file and avoid duplicate canonical names.
                df = df.drop(columns=["date_iso"])
            df = df.rename(columns=self.RAW_COL_MAP)
            if "date_iso" not in df.columns and "timestamp" in df.columns:
                df["date_iso"] = df["timestamp"]
            if "date_iso" not in df.columns:
                raise ValueError(
                    f"Missing date column in {csv_path.name}. "
                    "Expected one of: Date/Time, Date/Time (LST), date_iso, timestamp."
                )

            df = df.copy()
            df["date_iso"] = pd.to_datetime(df["date_iso"], errors="coerce")
            df = df.dropna(subset=["date_iso"]).sort_values("date_iso")

            if not set(self.DATE_FEATURE_COLS).issubset(df.columns):
                day_of_year = df["date_iso"].dt.dayofyear
                days_in_year = np.where(df["date_iso"].dt.is_leap_year, 366.0, 365.0)

                generated = {
                    "year_num": df["date_iso"].dt.year,
                    "doy_sin": np.sin(2.0 * np.pi * (day_of_year - 1) / days_in_year),
                    "doy_cos": np.cos(2.0 * np.pi * (day_of_year - 1) / days_in_year),
                }
                for col in self.DATE_FEATURE_COLS:
                    if col not in df.columns:
                        df[col] = generated[col]

            missing = sorted(required_cols - set(df.columns))
            if missing:
                raise ValueError(f"Missing columns in {csv_path.name}: {missing}")

            for col in set(self.FEATURE_COLS + self.TARGET_COLS):
                df[col] = pd.to_numeric(df[col], errors="coerce")

            # Targets must exist for supervised next-day prediction.
            df = df.dropna(subset=self.TARGET_COLS)

            # Fill feature gaps to preserve sequence continuity.
            for col in self.FEATURE_COLS:
                series = df[col]
                if series.isna().all():
                    df[col] = 0.0
                    continue
                series = series.interpolate(limit_direction="both")
                median_val = float(series.median()) if not pd.isna(series.median()) else 0.0
                df[col] = series.fillna(median_val).fillna(0.0)

            df = df[selected_cols].reset_index(drop=True)
            if not df.empty:
                station_frames.append(df)

        if not station_frames:
            raise ValueError(f"No valid station CSVs found in {self.data_dir}")

        combined_df = pd.concat(station_frames, ignore_index=True)
        self.feature_cols = list(self.FEATURE_COLS)
        self.target_cols = list(self.TARGET_COLS)
        self.input_size = len(self.feature_cols)
        self.output_size = len(self.target_cols)

        features = combined_df[self.feature_cols].to_numpy(dtype=float)
        targets = combined_df[self.target_cols].to_numpy(dtype=float)

        self.x_scaler = MinMaxScaler(feature_range=feature_range)
        self.y_scaler = MinMaxScaler(feature_range=feature_range)
        self.x_scaler.fit(features)
        self.y_scaler.fit(targets)

        xs = []
        ys = []
        for df in station_frames:
            raw_features = df[self.feature_cols].to_numpy(dtype=float)
            raw_targets = df[self.target_cols].to_numpy(dtype=float)
            scaled_features = self.x_scaler.transform(raw_features)
            scaled_targets = self.y_scaler.transform(raw_targets)

            n_rows = len(df)
            for i in range(n_rows - self.seq_len - self.horizon + 1):
                xs.append(scaled_features[i : i + self.seq_len])
                ys.append(scaled_targets[i + self.seq_len + self.horizon - 1])

        if not xs:
            raise ValueError(
                f"No sequence windows created. Check seq_len={self.seq_len} and horizon={self.horizon}."
            )

        self.x = torch.tensor(np.array(xs), dtype=self.dtype)
        self.y = torch.tensor(np.array(ys), dtype=self.dtype)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def inverse_transform_y(self, y_tensor: torch.Tensor) -> np.ndarray:
        y_np = y_tensor.detach().cpu().numpy()
        if y_np.ndim == 1:
            y_np = y_np.reshape(-1, self.output_size)
        elif y_np.ndim > 2:
            y_np = y_np.reshape(-1, self.output_size)
        return self.y_scaler.inverse_transform(y_np)
