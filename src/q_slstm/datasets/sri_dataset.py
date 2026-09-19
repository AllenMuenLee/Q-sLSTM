
"""
Sri Lanka Climate–Disease Dataset Loader
========================================

Features:
- Load & clean the CSV (drop index-like cols, parse time)
- Optional region filter or "ALL" aggregation
- Time-based split (train/val/test)
- Feature engineering: lags & rolling means
- Scaling: StandardScaler (fit only on train), encodes region (one-hot)
- Returns NumPy arrays and PyTorch Datasets for tabular/sequence tasks

Author: ChatGPT (for NASA_Sri_Lanka project)
"""

from typing import List, Tuple, Optional, Dict, Any
import os, pickle
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


DEFAULT_TARGET = "cases"

# ---------------------------
# Loading & Cleaning
# ---------------------------


def load_raw(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Drop likely index columns if present
    for c in ["Unnamed: 0", "index"]:
        if c in df.columns:
            df = df.drop(columns=[c])
    # Parse / construct time
    if "time" in df.columns:
        # If numeric weekly index like 315..790 (2013-2022 weekly)
        if pd.api.types.is_numeric_dtype(df["time"]):
            tmin, tmax = df["time"].min(), df["time"].max()
            # Heuristic: treat as weekly index
            if 300 <= tmin <= 10000 and 300 <= tmax <= 10000 and df["time"].nunique() > 100:
                base = pd.Timestamp('2013-01-01')
                df['time_index'] = df['time'].astype(int)
                # Align the minimum index to base; step=7 days per index increment
                df['time'] = (df['time_index'] - int(tmin)).apply(lambda k: base + pd.Timedelta(days=int(k * 7)))
            else:
                df["time"] = pd.to_datetime(df["time"], errors="coerce", infer_datetime_format=True)
        else:
            df["time"] = pd.to_datetime(df["time"], errors="coerce", infer_datetime_format=True)
    else:
        raise ValueError("CSV must contain a 'time' column")
    # Sort for determinism
    sort_cols = [c for c in ["region", "time"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    return df



# ---------------------------
# Feature Engineering
# ---------------------------

def add_lag_features(
    df: pd.DataFrame,
    group_col: str = "region",
    target: str = DEFAULT_TARGET,
    lags: List[int] = [1, 2, 4, 8, 12, 24],
    rollings: List[int] = [4, 12, 24],
    rolling_features: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Adds target lags and rolling means for selected features, grouped by region.
    Assumes data is sorted by [group_col, time].
    """
    df = df.copy()
    if rolling_features is None:
        # By default, apply rolling means to all numeric features except target
        rolling_features = [c for c in df.select_dtypes(include=[np.number]).columns if c != target]
    g = df.groupby(group_col, group_keys=False) if group_col in df.columns else [(None, df)]
    out = []
    for _, gdf in g:
        gdf = gdf.copy()
        # Lags of target
        for L in lags:
            gdf[f"{target}_lag{L}"] = gdf[target].shift(L)
        # Rolling means for features
        for w in rollings:
            for f in rolling_features:
                gdf[f"{f}_roll{w}"] = gdf[f].rolling(window=w, min_periods=max(1, w//2)).mean()
        out.append(gdf)
    df2 = pd.concat(out, axis=0).sort_values(["region","time"] if "region" in df.columns else "time").reset_index(drop=True)
    return df2


# ---------------------------
# Train / Val / Test split by time
# ---------------------------

def split_by_time(
    df: pd.DataFrame,
    train_end: str = "2020-12-31",
    val_end: str = "2021-12-31"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not np.issubdtype(df["time"].dtype, np.datetime64):
        raise ValueError("'time' must be datetime64 dtype")
    tr = df[df["time"] <= pd.to_datetime(train_end)].copy()
    va = df[(df["time"] > pd.to_datetime(train_end)) & (df["time"] <= pd.to_datetime(val_end))].copy()
    te = df[df["time"] > pd.to_datetime(val_end)].copy()
    return tr, va, te


# ---------------------------
# Preprocessing (scaling & encoding)
# ---------------------------

def build_feature_matrix(
    df: pd.DataFrame,
    target: str = DEFAULT_TARGET,
    drop_cols: Optional[List[str]] = None,
    region_encoding: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Builds X, y, feature_names from a cleaned DataFrame.
    - Drops 'time' and target from features
    - Optionally one-hot encodes 'region'
    """
    if drop_cols is None:
        drop_cols = []
    use_df = df.copy()
    # Separate target
    if target not in use_df.columns:
        raise ValueError(f"target '{target}' not in df")
    y = use_df[target].to_numpy(dtype=float)

    # Prepare features
    feat_df = use_df.drop(columns=[target])
    # Drop non-feature columns
    for c in ["time"] + drop_cols:
        if c in feat_df.columns:
            feat_df = feat_df.drop(columns=[c])

    # Handle region encoding if present
    feature_names = []
    if region_encoding and "region" in use_df.columns:
        # Separate region
        region_col = use_df["region"].astype(str).to_numpy().reshape(-1, 1)
        ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
        region_X = ohe.fit_transform(region_col)
        region_names = [f"region={cat}" for cat in ohe.categories_[0].tolist()]
        # Remove original region from features if still present
        if "region" in feat_df.columns:
            feat_df = feat_df.drop(columns=["region"])
        # Numeric features
        num_X = feat_df.select_dtypes(include=[np.number]).to_numpy(dtype=float)
        num_names = feat_df.select_dtypes(include=[np.number]).columns.tolist()
        X = np.hstack([num_X, region_X])
        feature_names = num_names + region_names
        enc = {"type": "onehot", "categories": ohe.categories_[0].tolist()}
    else:
        # Only numeric
        X = feat_df.select_dtypes(include=[np.number]).to_numpy(dtype=float)
        feature_names = feat_df.select_dtypes(include=[np.number]).columns.tolist()
        enc = None

    return X, y, feature_names


# def fit_scalers(
#     X_tr: np.ndarray, y_tr: np.ndarray,
#     scale_y: bool = True
# ) -> Dict[str, Any]:
#     x_scaler = StandardScaler().fit(X_tr)
#     y_scaler = StandardScaler().fit(y_tr.reshape(-1,1)) if scale_y else None
#     return {"x_scaler": x_scaler, "y_scaler": y_scaler}

def fit_scalers(
    X_tr: np.ndarray, y_tr: np.ndarray,
    y_scaler_type: str = "standard"  # "standard" | "minmax" | "none"
) -> Dict[str, Any]:
    x_scaler = StandardScaler().fit(X_tr)
    if y_scaler_type == "standard":
        y_scaler = StandardScaler().fit(y_tr.reshape(-1,1))
    elif y_scaler_type == "minmax":
        y_scaler = MinMaxScaler().fit(y_tr.reshape(-1,1))
    else:  # "none"
        y_scaler = None
    return {"x_scaler": x_scaler, "y_scaler": y_scaler}


def apply_scalers(
    X: np.ndarray, y: Optional[np.ndarray],
    scalers: Dict[str, Any]
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    Xs = scalers["x_scaler"].transform(X)
    if y is not None and scalers.get("y_scaler", None) is not None:
        ys = scalers["y_scaler"].transform(y.reshape(-1,1)).reshape(-1)
    else:
        ys = y
    return Xs, ys


# ---------------------------
# PyTorch Datasets
# ---------------------------

if TORCH_AVAILABLE:
    class TabularDataset(Dataset):
        def __init__(self, X: np.ndarray, y: np.ndarray):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)

        def __len__(self):
            return self.X.shape[0]

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    class SeqDataset(Dataset):
        """
        Builds (window, target) sequences per region.
        - window_len: input timesteps length
        - horizon: predict y at t + horizon (default 1)
        - features: numeric features (optionally include past 'cases')
        """
        def __init__(
            self,
            df: pd.DataFrame,
            feature_cols: List[str],
            target: str = DEFAULT_TARGET,
            group_col: str = "region",
            window_len: int = 12,
            horizon: int = 1,
            y_scaler_type: str = "standard",  # "standard" | "minmax" | "none"
            scalers: Optional[Dict[str, Any]] = None
        ):
            if "time" not in df.columns:
                raise ValueError("df must include 'time'")
            self.target = target
            self.window_len = window_len
            self.horizon = horizon
            self.group_col = group_col

            # Sort
            order_cols = [c for c in [group_col, "time"] if c in df.columns]
            gdf = df.sort_values(order_cols).reset_index(drop=True)

            # Build sequences per group
            self.X_list, self.y_list = [], []
            groups = [gdf] if group_col not in gdf.columns else [g for _, g in gdf.groupby(group_col)]
            for g in groups:
                g = g.reset_index(drop=True)
                feat = g[feature_cols].to_numpy(dtype=float)
                tgt = g[target].to_numpy(dtype=float)
                T = len(g)
                for t in range(T - window_len - horizon + 1):
                    x_win = feat[t : t + window_len]
                    y_t = tgt[t + window_len + horizon - 1]
                    self.X_list.append(x_win)
                    self.y_list.append(y_t)

            X = np.stack(self.X_list, axis=0) if self.X_list else np.empty((0, window_len, len(feature_cols)))
            y = np.array(self.y_list, dtype=float) if self.y_list else np.empty((0,))

            # Flatten for scaling (fit beforehand and pass scalers, or fit here locally)
            X_2d = X.reshape(X.shape[0], -1) if X.size > 0 else X
            if scalers is None and X_2d.size > 0:
                x_scaler = StandardScaler().fit(X_2d)
                # y_scaler = StandardScaler().fit(y.reshape(-1,1))
                if y_scaler_type == "standard":
                    # print("Use standard scaler")
                    y_scaler = StandardScaler().fit(y.reshape(-1,1))
                elif y_scaler_type == "minmax":
                    # print("Use minmax scaler")
                    y_scaler = MinMaxScaler().fit(y.reshape(-1,1))
                else:
                    y_scaler = None
                self.scalers = {"x_scaler": x_scaler, "y_scaler": y_scaler}
            else:
                self.scalers = scalers

            if X_2d.size > 0:
                Xs = self.scalers["x_scaler"].transform(X_2d).reshape(X.shape[0], X.shape[1], X.shape[2])
                if self.scalers.get("y_scaler"):
                    ys = self.scalers["y_scaler"].transform(y.reshape(-1,1)).reshape(-1)
                else:
                    ys = y
                # Xs = self.scalers["x_scaler"].transform(X_2d).reshape(X.shape[0], X.shape[1], X.shape[2])
                # ys = self.scalers["y_scaler"].transform(y.reshape(-1,1)).reshape(-1) if self.scalers.get("y_scaler") else y
            else:
                Xs, ys = X, y

            self.X = torch.tensor(Xs, dtype=torch.float32)
            self.y = torch.tensor(ys, dtype=torch.float32)

        def __len__(self):
            return self.X.shape[0]

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]


# ---------------------------
# High-level convenience
# ---------------------------

def prepare_tabular(
    csv_path: str,
    target: str = DEFAULT_TARGET,
    add_lags_flag: bool = True,
    lags: List[int] = [1,2,4,8,12,24],
    rolling_windows: List[int] = [4,12,24],
    train_end: str = "2020-12-31",
    val_end: str = "2021-12-31",
    y_scaler_type: str = "standard",  # "standard" | "minmax" | "none"
    scale_y: Optional[bool] = None
) -> Dict[str, Any]:
    """
    Returns dict with train/val/test (X, y), feature_names and scalers.
    """

    if scale_y is not None:
        y_scaler_type = "standard" if scale_y else "none"

    df = load_raw(csv_path)
    if add_lags_flag:
        df = add_lag_features(df, lags=lags, rollings=rolling_windows, target=target)

    tr, va, te = split_by_time(df, train_end, val_end)
    X_tr, y_tr, feat_names = build_feature_matrix(tr, target=target)
    X_va, y_va, _ = build_feature_matrix(va, target=target)
    X_te, y_te, _ = build_feature_matrix(te, target=target)

    # scalers = fit_scalers(X_tr, y_tr, scale_y=scale_y)
    scalers = fit_scalers(X_tr, y_tr, y_scaler_type=y_scaler_type)
    X_trs, y_trs = apply_scalers(X_tr, y_tr, scalers)
    X_vas, y_vas = apply_scalers(X_va, y_va, scalers)
    X_tes, y_tes = apply_scalers(X_te, y_te, scalers)

    return {
        "X_tr": X_trs, "y_tr": y_trs,
        "X_va": X_vas, "y_va": y_vas,
        "X_te": X_tes, "y_te": y_tes,
        "feature_names": feat_names,
        "scalers": scalers,
        "splits": {"train_end": train_end, "val_end": val_end}
    }


def prepare_sequence(
    csv_path: str,
    target: str = DEFAULT_TARGET,
    feature_cols: Optional[List[str]] = None,
    window_len: int = 12,
    horizon: int = 1,
    train_end: str = "2020-12-31",
    val_end: str = "2021-12-31",
    y_scaler_type: str = "standard"
) -> Dict[str, Any]:
    """
    Returns dict with PyTorch SeqDataset for train/val/test (if torch available), else numpy.
    """
    df = load_raw(csv_path)
    # default features: all numeric except target
    if feature_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c != target]

    tr, va, te = split_by_time(df, train_end, val_end)

    if TORCH_AVAILABLE:
        # Fit scalers on train sequences
        tmp_tr = SeqDataset(tr, feature_cols, target=target, window_len=window_len, horizon=horizon, scalers=None,y_scaler_type=y_scaler_type)
        scalers = tmp_tr.scalers
        ds_tr = SeqDataset(tr, feature_cols, target=target, window_len=window_len, horizon=horizon, scalers=scalers, y_scaler_type=y_scaler_type)
        ds_va = SeqDataset(va, feature_cols, target=target, window_len=window_len, horizon=horizon, scalers=scalers, y_scaler_type=y_scaler_type)
        ds_te = SeqDataset(te, feature_cols, target=target, window_len=window_len, horizon=horizon, scalers=scalers, y_scaler_type=y_scaler_type)
        return {"train": ds_tr, "val": ds_va, "test": ds_te, "scalers": scalers, "feature_cols": feature_cols}
    else:
        # Numpy fallback
        def to_np(df_part):
            # Simple rolling window without scaling
            df_part = df_part.sort_values(["region","time"] if "region" in df_part.columns else "time")
            Xs, ys = [], []
            groups = [df_part] if "region" not in df_part.columns else [g for _, g in df_part.groupby("region")]
            for g in groups:
                g = g.reset_index(drop=True)
                feat = g[feature_cols].to_numpy(dtype=float)
                tgt = g[target].to_numpy(dtype=float)
                T = len(g)
                for t in range(T - window_len - horizon + 1):
                    Xs.append(feat[t:t+window_len])
                    ys.append(tgt[t + window_len + horizon - 1])
            return np.array(Xs), np.array(ys)

        Xtr, ytr = to_np(tr); Xva, yva = to_np(va); Xte, yte = to_np(te)
        return {"X_tr": Xtr, "y_tr": ytr, "X_va": Xva, "y_va": yva, "X_te": Xte, "y_te": yte, "feature_cols": feature_cols}


# ---------------------------
# Save / Load scalers (for inference reproducibility)
# ---------------------------

def save_scalers(scalers: Dict[str, Any], path: str):
    with open(path, "wb") as f:
        pickle.dump(scalers, f)

def load_scalers(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------
# Region utilities
# ---------------------------

def list_regions(csv_path: str) -> list:
    """Return sorted unique region names from the CSV."""
    df = load_raw(csv_path)
    if "region" not in df.columns:
        return []
    return sorted(df["region"].astype(str).unique().tolist())

def filter_region(df: pd.DataFrame, region_name: str) -> pd.DataFrame:
    """Return a copy of df filtered to a single region (exact match)."""
    if "region" not in df.columns:
        raise ValueError("No 'region' column in DataFrame")
    m = df["region"].astype(str) == str(region_name)
    out = df.loc[m].copy().reset_index(drop=True)
    if out.empty:
        raise ValueError(f"Region '{region_name}' not found or has no rows.")
    return out

def prepare_tabular_region(
    csv_path: str,
    region_name: str,
    target: str = DEFAULT_TARGET,
    add_lags_flag: bool = True,
    lags: List[int] = [1,2,4,8,12,24],
    rolling_windows: List[int] = [4,12,24],
    train_end: str = "2020-12-31",
    val_end: str = "2021-12-31",
    y_scaler_type: str = "standard",  # "standard" | "minmax" | "none"
    scale_y: Optional[bool] = None,
    one_hot_region: bool = False
) -> Dict[str, Any]:
    """
    Same as prepare_tabular, but restricted to one region.
    By default, one_hot_region=False because it's a single category.
    """
    df = load_raw(csv_path)
    df = filter_region(df, region_name)
    if add_lags_flag:
        df = add_lag_features(df, lags=lags, rollings=rolling_windows, target=target)

    tr, va, te = split_by_time(df, train_end, val_end)

    # region_encoding toggled
    def build(Xdf):
        X, y, names = build_feature_matrix(
            Xdf, target=target, drop_cols=None, region_encoding=one_hot_region
        )
        return X, y, names

    X_tr, y_tr, feat_names = build(tr)
    X_va, y_va, _ = build(va)
    X_te, y_te, _ = build(te)

    # scalers = fit_scalers(X_tr, y_tr, scale_y=scale_y)
    scalers = fit_scalers(X_tr, y_tr, y_scaler_type=y_scaler_type)
    X_trs, y_trs = apply_scalers(X_tr, y_tr, scalers)
    X_vas, y_vas = apply_scalers(X_va, y_va, scalers)
    X_tes, y_tes = apply_scalers(X_te, y_te, scalers)

    return {
        "region": region_name,
        "X_tr": X_trs, "y_tr": y_trs,
        "X_va": X_vas, "y_va": y_vas,
        "X_te": X_tes, "y_te": y_tes,
        "feature_names": feat_names,
        "scalers": scalers,
        "splits": {"train_end": train_end, "val_end": val_end}
    }

def prepare_sequence_region(
    csv_path: str,
    region_name: str,
    target: str = DEFAULT_TARGET,
    feature_cols: Optional[List[str]] = None,
    window_len: int = 12,
    horizon: int = 1,
    train_end: str = "2020-12-31",
    val_end: str = "2021-12-31",
    y_scaler_type: str = "standard"
) -> Dict[str, Any]:
    """
    Same as prepare_sequence, but restricted to one region.
    """
    df = load_raw(csv_path)
    df = filter_region(df, region_name)

    # default features: all numeric except target
    if feature_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c != target]

    tr, va, te = split_by_time(df, train_end, val_end)

    if TORCH_AVAILABLE:
        tmp_tr = SeqDataset(tr, feature_cols, target=target, window_len=window_len, horizon=horizon, scalers=None, y_scaler_type=y_scaler_type)
        scalers = tmp_tr.scalers
        ds_tr = SeqDataset(tr, feature_cols, target=target, window_len=window_len, horizon=horizon, scalers=scalers, y_scaler_type=y_scaler_type)
        ds_va = SeqDataset(va, feature_cols, target=target, window_len=window_len, horizon=horizon, scalers=scalers, y_scaler_type=y_scaler_type)
        ds_te = SeqDataset(te, feature_cols, target=target, window_len=window_len, horizon=horizon, scalers=scalers, y_scaler_type=y_scaler_type)
        return {"region": region_name, "train": ds_tr, "val": ds_va, "test": ds_te, "scalers": scalers, "feature_cols": feature_cols}
    else:
        # Numpy fallback
        def to_np(df_part):
            df_part = df_part.sort_values(["time"])
            Xs, ys = [], []
            g = df_part.reset_index(drop=True)
            feat = g[feature_cols].to_numpy(dtype=float)
            tgt = g[target].to_numpy(dtype=float)
            T = len(g)
            for t in range(T - window_len - horizon + 1):
                Xs.append(feat[t:t+window_len])
                ys.append(tgt[t + window_len + horizon - 1])
            return np.array(Xs), np.array(ys)

        Xtr, ytr = to_np(tr); Xva, yva = to_np(va); Xte, yte = to_np(te)
        return {"region": region_name, "X_tr": Xtr, "y_tr": ytr, "X_va": Xva, "y_va": yva, "X_te": Xte, "y_te": yte, "feature_cols": feature_cols}

def save_region_subset(csv_path: str, region_name: str, out_path: str) -> str:
    """Save a single-region subset CSV for quick inspection or external tools."""
    df = load_raw(csv_path)
    sub = filter_region(df, region_name)
    sub.to_csv(out_path, index=False)
    return out_path
