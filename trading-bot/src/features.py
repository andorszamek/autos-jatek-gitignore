"""Feature engineering — single source of truth for training and inference.

All features are computed from past data only (no look-ahead bias).
FEATURE_COLUMNS is the canonical list used by both training and inference.
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FEATURE_COLUMNS: list[str] = [
    "SMA_5",
    "SMA_10",
    "SMA_20",
    "RSI_14",
    "VOL_20",
    "RET_1",
    "RET_5",
    "RET_10",
    "VOL_RATIO",
]

TARGET_COLUMN = "next_day_up"


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Standard Wilder RSI — no look-ahead bias.

    Uses exponential moving average of gains/losses (alpha = 1/period).
    Values are always in [0, 100].
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    # Use exponential weighted mean (Wilder smoothing: alpha = 1/period)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # When avg_loss == 0 and avg_gain > 0 → RSI = 100
    rsi = rsi.where(avg_loss != 0, other=100.0)
    return rsi


def _build_symbol_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute features for a single-symbol DataFrame (must be sorted by timestamp ascending)."""
    out = pd.DataFrame(index=df.index)

    close = df["close"]
    volume = df["volume"]

    # SMA
    out["SMA_5"] = close.rolling(5, min_periods=5).mean()
    out["SMA_10"] = close.rolling(10, min_periods=10).mean()
    out["SMA_20"] = close.rolling(20, min_periods=20).mean()

    # RSI (no look-ahead — uses only past data)
    out["RSI_14"] = _rsi(close, period=14)

    # Volatility: 20-day rolling std of log returns
    log_ret = np.log(close / close.shift(1))
    out["VOL_20"] = log_ret.rolling(20, min_periods=20).std()

    # Returns
    out["RET_1"] = close.pct_change(1)
    out["RET_5"] = close.pct_change(5)
    out["RET_10"] = close.pct_change(10)

    # Volume ratio
    vol_ma20 = volume.rolling(20, min_periods=20).mean()
    out["VOL_RATIO"] = volume / vol_ma20

    # Target: next_day_up = whether tomorrow's close > today's close
    # shift(-1) means: at row i, target = (close[i+1] > close[i])
    # Explicitly propagate NaN for the last row: NaN > close evaluates to False
    # in NumPy (not NaN), which would give a spurious target=0 for the last row.
    next_close = close.shift(-1)
    out[TARGET_COLUMN] = np.where(next_close.notna(), (next_close > close).astype(float), np.nan)

    return out


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    """Compute features and target from canonical OHLCV DataFrame.

    Returns:
        (X, y) where:
          - X: pd.DataFrame with FEATURE_COLUMNS
          - y: pd.Series of int {0, 1} targets, or None if target cannot be computed
            (e.g. inference mode with only the last row available)

    Notes:
        - Rows with NaN features are dropped.
        - The last row always has a NaN target (no tomorrow data), so it is dropped
          from training. In inference mode, only the last row matters; y is None.
    """
    if df.empty:
        raise ValueError("[features] Input DataFrame is empty.")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    # Build features per symbol, then combine
    symbol_frames: list[pd.DataFrame] = []
    symbols = df["symbol"].unique()

    for sym in symbols:
        sym_mask = df["symbol"] == sym
        sym_df = df[sym_mask].copy()
        feat_df = _build_symbol_features(sym_df)

        # Bring back identifying columns
        feat_df["symbol"] = sym
        feat_df["timestamp"] = sym_df["timestamp"].values
        symbol_frames.append(feat_df)

    feat_all = pd.concat(symbol_frames, axis=0)
    feat_all = feat_all.reset_index(drop=True)

    # Check if we can form a target
    has_valid_target = feat_all[TARGET_COLUMN].notna().any()

    if not has_valid_target:
        # Inference mode: return X without target
        X = feat_all[FEATURE_COLUMNS].copy()
        # Drop rows with NaN features
        X = X.dropna(subset=FEATURE_COLUMNS)
        if X.empty:
            raise ValueError("[features] No valid feature rows after dropping NaNs.")
        logger.debug("[features] Inference mode: returning X only (no target).")
        return X, None

    # Training mode: drop rows where features OR target are NaN
    feat_all[TARGET_COLUMN] = feat_all[TARGET_COLUMN].astype("Int64")

    drop_mask = feat_all[FEATURE_COLUMNS + [TARGET_COLUMN]].isna().any(axis=1)
    feat_clean = feat_all[~drop_mask].reset_index(drop=True)

    if feat_clean.empty:
        raise ValueError("[features] No valid rows after dropping NaN features and target.")

    X = feat_clean[FEATURE_COLUMNS].copy()
    y = feat_clean[TARGET_COLUMN].astype(int).copy()

    logger.info(
        "[features] Built %d samples, %d features. Target distribution: up=%.1f%%",
        len(X),
        len(FEATURE_COLUMNS),
        100.0 * y.mean(),
    )

    return X, y
