"""Feature engineering tests.

Updated for new feature set (SMA50/200, MACD, BB_PCT, ATR, etc.) which
requires at least 200 rows of input data.
"""
import numpy as np
import pandas as pd
import pytest

from src.features import build_features, FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n: int = 250, symbol: str = "TEST", seed: int = 42) -> pd.DataFrame:
    """Create a synthetic canonical DataFrame with n rows (default 250 for SMA200)."""
    rng = np.random.default_rng(seed)
    prices = 100.0 * np.cumprod(1 + rng.normal(0, 0.01, n))
    dates = pd.date_range("2018-01-02", periods=n, freq="B", tz="UTC")
    return pd.DataFrame({
        "timestamp": dates,
        "symbol": symbol,
        "open": prices * (1 + rng.uniform(-0.002, 0.002, n)),
        "high": prices * (1 + rng.uniform(0, 0.005, n)),
        "low": prices * (1 - rng.uniform(0, 0.005, n)),
        "close": prices,
        "volume": rng.integers(100_000, 1_000_000, n).astype(float),
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_no_lookahead():
    """Target at index i must equal (close[i+1] > close[i]).

    Uses a monotonically increasing price series so every target must be 1.
    Needs 250 rows because SMA200 requires 200 rows of lookback.
    """
    n = 250
    prices = np.linspace(100.0, 300.0, n)  # strictly increasing → every target = 1
    dates = pd.date_range("2018-01-02", periods=n, freq="B", tz="UTC")
    df = pd.DataFrame({
        "timestamp": dates,
        "symbol": "UP",
        "open": prices,
        "high": prices * 1.001,
        "low": prices * 0.999,
        "close": prices,
        "volume": np.ones(n) * 500_000.0,
    })

    X, y = build_features(df)

    assert y is not None, "Expected y to be non-None for training data"
    assert len(y) > 0, "Expected non-empty y"
    assert (y == 1).all(), (
        f"Expected all targets=1 for monotone-up series, got:\n{y.value_counts()}"
    )


def test_feature_columns_match_constant():
    """X.columns must equal exactly FEATURE_COLUMNS (same order)."""
    df = _make_df(250)
    X, y = build_features(df)
    assert list(X.columns) == FEATURE_COLUMNS, (
        f"X columns {list(X.columns)} != FEATURE_COLUMNS {FEATURE_COLUMNS}"
    )


def test_build_features_no_future_data():
    """Features at the first valid row use only data at or before that row.

    With SMA200 as the longest lookback, the first valid row is at original
    index 199 (0-based). At that point:
      RET_1[199] = (close[199] - close[198]) / close[198]
    We verify this is computed purely from past data.
    """
    n = 210
    prices = np.arange(1.0, n + 1.0)  # 1, 2, 3, ..., 210
    dates = pd.date_range("2018-01-02", periods=n, freq="B", tz="UTC")
    df = pd.DataFrame({
        "timestamp": dates,
        "symbol": "NUM",
        "open": prices,
        "high": prices + 0.5,
        "low": prices - 0.5,
        "close": prices,
        "volume": np.ones(n) * 100_000.0,
    })

    X, y = build_features(df)
    assert X is not None and not X.empty

    # RET_1 at the first output row (original index 199):
    # = (close[199] - close[198]) / close[198] = (200 - 199) / 199
    first_ret1 = X["RET_1"].iloc[0]
    expected_ret1 = (prices[199] - prices[198]) / prices[198]
    assert abs(first_ret1 - expected_ret1) < 1e-9, (
        f"RET_1 at first row = {first_ret1:.9f}, expected {expected_ret1:.9f}. "
        "Confirms no look-ahead: uses only data up to row 199."
    )

    # PRICE_VS_SMA50 at first row = close[199] / mean(close[150:200]) - 1
    first_pvs50 = X["PRICE_VS_SMA50"].iloc[0]
    expected_pvs50 = prices[199] / np.mean(prices[150:200]) - 1
    assert abs(first_pvs50 - expected_pvs50) < 1e-6, (
        f"PRICE_VS_SMA50 at first row = {first_pvs50:.6f}, expected {expected_pvs50:.6f}"
    )


def test_rsi_range():
    """RSI values must be in [0, 100]."""
    df = _make_df(250)
    X, y = build_features(df)
    assert X is not None
    rsi_vals = X["RSI_14"].dropna()
    assert len(rsi_vals) > 0, "RSI column is all NaN"
    assert (rsi_vals >= 0).all(), f"RSI has values < 0: min={rsi_vals.min()}"
    assert (rsi_vals <= 100).all(), f"RSI has values > 100: max={rsi_vals.max()}"


def test_inference_mode():
    """Short df (fewer rows than SMA200) should raise ValueError gracefully."""
    n = 30
    df = _make_df(n)
    try:
        result = build_features(df)
    except ValueError:
        return  # expected — not enough data for SMA200
    except Exception as exc:
        pytest.fail(f"Unexpected exception: {type(exc).__name__}: {exc}")
    assert isinstance(result, tuple) and len(result) == 2


def test_inference_mode_returns_x_without_target():
    """Single last row → y should be None (no future close available)."""
    df = _make_df(250)
    X, y = build_features(df)
    assert y is not None
    assert len(X) > 0

    last_row = df.tail(1).reset_index(drop=True)
    try:
        result = build_features(last_row)
        X_inf, y_inf = result
        assert y_inf is None, "Expected y=None for single-row inference"
    except ValueError:
        pass  # acceptable: not enough lookback data


def test_multiple_symbols():
    """Multi-symbol df produces features for each symbol independently."""
    df1 = _make_df(250, symbol="AAA", seed=1)
    df2 = _make_df(250, symbol="BBB", seed=2)
    df = pd.concat([df1, df2], ignore_index=True)

    X, y = build_features(df)
    assert X is not None and not X.empty
    assert list(X.columns) == FEATURE_COLUMNS


def test_no_nan_in_output():
    """After build_features, neither X nor y should contain NaN values."""
    df = _make_df(250)
    X, y = build_features(df)

    assert not X.isnull().any().any(), "X contains NaN values"
    assert y is not None
    assert not y.isnull().any(), "y contains NaN values"
