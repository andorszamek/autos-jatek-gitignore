"""Feature engineering — single source of truth for training and inference.

All features are computed from past data only (no look-ahead bias).
FEATURE_COLUMNS is the canonical list used by both training and inference.
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FEATURE_COLUMNS: list[str] = [
    # Trend
    "PRICE_VS_SMA50",    # how far price is above/below 50-day MA
    "PRICE_VS_SMA200",   # trend filter: >0 = bull regime, <0 = bear
    "SMA50_VS_SMA200",   # golden/death cross signal
    # Momentum
    "RET_1",
    "RET_5",
    "RET_20",
    "RET_60",            # 3-month momentum
    # Mean reversion
    "RSI_14",
    "BB_PCT",            # Bollinger %B: where price sits in the band
    # MACD
    "MACD_HIST",         # MACD histogram (signal divergence)
    # Volatility / regime
    "ATR_PCT",           # ATR as % of price (normalised)
    "VOL_20",            # rolling volatility
    # Volume
    "VOL_RATIO",
]

TARGET_COLUMN = "next_day_up"


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Standard Wilder RSI — no look-ahead bias."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.where(avg_loss != 0, other=100.0)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range — no look-ahead bias."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def _build_symbol_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute features for a single-symbol DataFrame (sorted ascending by timestamp)."""
    out = pd.DataFrame(index=df.index)

    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]

    # --- Trend features ---
    sma50  = close.rolling(50,  min_periods=50).mean()
    sma200 = close.rolling(200, min_periods=200).mean()
    out["PRICE_VS_SMA50"]  = (close / sma50 - 1)
    out["PRICE_VS_SMA200"] = (close / sma200 - 1)
    out["SMA50_VS_SMA200"] = (sma50 / sma200 - 1)

    # --- Momentum ---
    out["RET_1"]  = close.pct_change(1)
    out["RET_5"]  = close.pct_change(5)
    out["RET_20"] = close.pct_change(20)
    out["RET_60"] = close.pct_change(60)

    # --- RSI ---
    out["RSI_14"] = _rsi(close, period=14)

    # --- Bollinger %B: (price - lower) / (upper - lower) ---
    bb_mid   = close.rolling(20, min_periods=20).mean()
    bb_std   = close.rolling(20, min_periods=20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    band_width = (bb_upper - bb_lower).replace(0, np.nan)
    out["BB_PCT"] = (close - bb_lower) / band_width

    # --- MACD histogram (12-26 EMA, signal 9) ---
    ema12 = close.ewm(span=12, min_periods=12, adjust=False).mean()
    ema26 = close.ewm(span=26, min_periods=26, adjust=False).mean()
    macd_line   = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, min_periods=9, adjust=False).mean()
    out["MACD_HIST"] = (macd_line - macd_signal) / close  # normalised by price

    # --- ATR as % of price ---
    atr = _atr(high, low, close, period=14)
    out["ATR_PCT"] = atr / close

    # --- Volatility ---
    log_ret = np.log(close / close.shift(1))
    out["VOL_20"] = log_ret.rolling(20, min_periods=20).std()

    # --- Volume ratio ---
    vol_ma20 = volume.rolling(20, min_periods=20).mean()
    out["VOL_RATIO"] = volume / vol_ma20

    # --- Target: 5-day forward return (less noisy than 1-day) ---
    next_close = close.shift(-5)
    out[TARGET_COLUMN] = np.where(
        next_close.notna(),
        (next_close > close).astype(float),
        np.nan,
    )

    return out


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    """Compute features and target from canonical OHLCV DataFrame.

    Returns:
        (X, y) where X has FEATURE_COLUMNS and y is {0,1} or None in inference mode.
    """
    if df.empty:
        raise ValueError("[features] Input DataFrame is empty.")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    symbol_frames: list[pd.DataFrame] = []
    for sym in df["symbol"].unique():
        sym_df = df[df["symbol"] == sym].copy()
        feat_df = _build_symbol_features(sym_df)
        feat_df["symbol"]    = sym
        feat_df["timestamp"] = sym_df["timestamp"].values
        symbol_frames.append(feat_df)

    feat_all = pd.concat(symbol_frames, axis=0).reset_index(drop=True)
    has_valid_target = feat_all[TARGET_COLUMN].notna().any()

    if not has_valid_target:
        X = feat_all[FEATURE_COLUMNS].dropna(subset=FEATURE_COLUMNS)
        if X.empty:
            raise ValueError("[features] No valid feature rows after dropping NaNs.")
        logger.debug("[features] Inference mode: returning X only (no target).")
        return X, None

    feat_all[TARGET_COLUMN] = feat_all[TARGET_COLUMN].astype("Int64")
    drop_mask = feat_all[FEATURE_COLUMNS + [TARGET_COLUMN]].isna().any(axis=1)
    feat_clean = feat_all[~drop_mask].reset_index(drop=True)

    if feat_clean.empty:
        raise ValueError("[features] No valid rows after dropping NaN features and target.")

    X = feat_clean[FEATURE_COLUMNS].copy()
    y = feat_clean[TARGET_COLUMN].astype(int).copy()

    logger.info(
        "[features] Built %d samples, %d features. Target up=%.1f%%",
        len(X), len(FEATURE_COLUMNS), 100.0 * y.mean(),
    )
    return X, y
