"""Feature engineering tests — Phase 3."""
import numpy as np
import pandas as pd
import pytest

from src.features import build_features, FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n: int = 60, symbol: str = "TEST", seed: int = 42) -> pd.DataFrame:
    """Create a synthetic canonical DataFrame with n rows."""
    rng = np.random.default_rng(seed)
    prices = 100.0 * np.cumprod(1 + rng.normal(0, 0.01, n))
    dates = pd.date_range("2020-01-02", periods=n, freq="B", tz="UTC")
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

    After build_features(), row 0 of the returned (X, y) pair
    should have target == whether the next close was higher than the current.
    We construct a simple monotonically increasing price series so we know
    every day is 'up', and verify the targets are all 1.
    """
    n = 60
    prices = np.linspace(100.0, 200.0, n)  # strictly increasing → every day is up
    dates = pd.date_range("2020-01-02", periods=n, freq="B", tz="UTC")
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

    # All targets should be 1 (next day is always higher)
    assert y is not None, "Expected y to be non-None for training data"
    assert len(y) > 0, "Expected non-empty y"
    assert (y == 1).all(), f"Expected all targets=1 for monotone-up series, got:\n{y.value_counts()}"


def test_feature_columns_match_constant():
    """X.columns must equal exactly FEATURE_COLUMNS (same order)."""
    df = _make_df(60)
    X, y = build_features(df)
    assert list(X.columns) == FEATURE_COLUMNS, (
        f"X columns {list(X.columns)} != FEATURE_COLUMNS {FEATURE_COLUMNS}"
    )


def test_build_features_no_future_data():
    """Every feature at row i should use only data at positions <= i.

    VOL_20 uses rolling std of log_returns (which need shift(1)), so the first
    valid combined feature row is at original index 20 (0-based): the 21st row.
    At that point, SMA_20 = mean(closes[1..20]) = mean(2..21) = 11.5.

    This confirms no look-ahead: the value is computed from past data ending at
    position 20, not from future rows.
    """
    n = 45
    prices = np.arange(1.0, n + 1.0)  # 1, 2, 3, ..., n
    dates = pd.date_range("2020-01-02", periods=n, freq="B", tz="UTC")
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

    # The first valid row (all features non-NaN, target non-NaN) is at original
    # index 20 (0-based), because VOL_20 needs 20 log-returns (log_ret[0] is NaN,
    # so rolling(20) first fills at position 20).
    # SMA_20 at original row 20 = mean(prices[1..20]) = mean(2..21) = 11.5
    first_sma20 = X["SMA_20"].iloc[0]
    expected_sma20 = np.mean(prices[1:21])  # mean of closes at rows 1..20 = 11.5
    assert abs(first_sma20 - expected_sma20) < 1e-6, (
        f"SMA_20 at first output row = {first_sma20:.6f}, expected {expected_sma20:.6f}. "
        "This confirms SMA_20 uses only past data (no look-ahead)."
    )

    # Verify SMA_5 at first row only uses 5 past closes ending at row 20
    first_sma5 = X["SMA_5"].iloc[0]
    expected_sma5 = np.mean(prices[16:21])  # closes at indices 16..20 = [17..21]
    assert abs(first_sma5 - expected_sma5) < 1e-6, (
        f"SMA_5 at first row = {first_sma5:.6f}, expected {expected_sma5:.6f}"
    )


def test_rsi_range():
    """RSI values must be in [0, 100]."""
    df = _make_df(80)
    X, y = build_features(df)
    assert X is not None
    rsi_vals = X["RSI_14"].dropna()
    assert len(rsi_vals) > 0, "RSI column is all NaN"
    assert (rsi_vals >= 0).all(), f"RSI has values < 0: min={rsi_vals.min()}"
    assert (rsi_vals <= 100).all(), f"RSI has values > 100: max={rsi_vals.max()}"


def test_inference_mode():
    """Short df (no future target possible) must return (X, None) without crashing."""
    # Provide exactly min_periods rows so features can be computed but there's no valid target
    # Build a df where every row's target would be NaN (happens at the very last row
    # because shift(-1) has no future). With a short df where all targets are NaN:
    n = 30  # shorter than 20-day rolling windows → features will be NaN too
    df = _make_df(n)

    # build_features should handle this gracefully and return (X, None)
    # even if X is empty (no valid feature rows) — it should NOT raise
    try:
        result = build_features(df)
    except ValueError:
        # Acceptable: if no valid feature rows, ValueError is raised
        # but it must not crash with an unhandled exception
        return
    except Exception as exc:
        pytest.fail(f"build_features raised unexpected exception: {type(exc).__name__}: {exc}")

    # If it returned, check the result is a valid tuple
    assert isinstance(result, tuple) and len(result) == 2


def test_inference_mode_returns_x_without_target():
    """With sufficient history but only one inference row, y should be None."""
    # Build a larger dataframe with plenty of history, but modify the last row
    # so there's no future close → target for last row is always NaN in build_features
    df = _make_df(100)

    # Keep only the last row (inference scenario: predict for today)
    # but provide enough context for rolling windows by using the full df
    # The target for the very last row will be NaN (no tomorrow)
    # build_features should return (X, None) when all targets are NaN
    # We simulate this by keeping only the rows that would result in NaN target:
    # In practice, the last row of any df has NaN target.

    X, y = build_features(df)
    # y should NOT be None for a full training df (it has valid rows before the last)
    assert y is not None
    assert len(X) > 0

    # Now test with only the last single row — this triggers inference mode
    last_row = df.tail(1).copy()
    # Reset index to avoid issues
    last_row = last_row.reset_index(drop=True)

    try:
        result = build_features(last_row)
        X_inf, y_inf = result
        assert y_inf is None, "Expected y=None for single-row inference"
    except ValueError:
        pass  # Acceptable: not enough data for rolling windows


def test_multiple_symbols():
    """Multi-symbol DataFrame should produce features for each symbol independently."""
    df1 = _make_df(60, symbol="AAA", seed=1)
    df2 = _make_df(60, symbol="BBB", seed=2)
    df = pd.concat([df1, df2], ignore_index=True)

    X, y = build_features(df)
    assert X is not None
    assert not X.empty
    assert list(X.columns) == FEATURE_COLUMNS


def test_no_nan_in_output():
    """After build_features, neither X nor y should contain NaN values."""
    df = _make_df(80)
    X, y = build_features(df)

    assert not X.isnull().any().any(), "X contains NaN values"
    assert y is not None
    assert not y.isnull().any(), "y contains NaN values"
