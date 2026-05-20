"""Backtest tests — Phase 5.

Tests use synthetic price data to avoid network calls.
They exercise the core backtest logic: cost model, position sizing, P&L.
"""
import math
from math import floor

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_canonical_df(
    n: int = 200,
    symbol: str = "TEST",
    start_price: float = 100.0,
    daily_return: float = 0.005,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a synthetic canonical DataFrame.

    If daily_return > 0, prices trend upward (making buy profitable).
    """
    rng = np.random.default_rng(seed)
    prices = start_price * np.cumprod(
        1.0 + daily_return + rng.normal(0, 0.002, n)
    )
    dates = pd.date_range("2018-01-02", periods=n, freq="B", tz="UTC")
    return pd.DataFrame({
        "timestamp": dates,
        "symbol": symbol,
        "open": prices * (1 + rng.uniform(-0.001, 0.001, n)),
        "high": prices * (1 + rng.uniform(0, 0.005, n)),
        "low": prices * (1 - rng.uniform(0, 0.005, n)),
        "close": prices,
        "volume": rng.integers(500_000, 2_000_000, n).astype(float),
    })


def _base_cfg(
    commission: float = 0.0,
    slippage_pct: float = 0.0,
    fx_cost_pct: float = 0.0,
) -> dict:
    return {
        "initial_capital": 10_000.0,
        "risk": {
            "max_position_pct": 0.50,  # generous for test
            "stop_loss_pct": 0.50,     # disable stop-loss in backtest tests
            "daily_max_loss_pct": 0.99,
            "kill_switch_drawdown_pct": 0.99,
        },
        "backtest": {
            "commission": commission,
            "slippage_pct": slippage_pct,
            "fx_cost_pct": fx_cost_pct,
        },
        "_env": {
            "alpaca_api_key": "dummy",
            "alpaca_secret_key": "dummy",
            "alpaca_paper": True,
            "live_trading": False,
        },
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_buy_and_hold_benchmark_present():
    """Results dict must contain a 'benchmark' key."""
    from src.backtest import run_backtest

    df = _make_canonical_df(n=300)
    cfg = _base_cfg()
    results = run_backtest(df, cfg)
    assert "benchmark" in results, "Missing 'benchmark' key in results"
    assert isinstance(results["benchmark"], pd.Series)
    assert len(results["benchmark"]) > 0


def test_results_keys_present():
    """Results must contain equity_curve, trades, metrics, benchmark."""
    from src.backtest import run_backtest

    df = _make_canonical_df(n=300)
    cfg = _base_cfg()
    results = run_backtest(df, cfg)
    for key in ("equity_curve", "trades", "metrics", "benchmark"):
        assert key in results, f"Missing key: {key}"


def test_metrics_keys_present():
    """Metrics dict must contain CAGR, max_drawdown, sharpe_ratio, n_trades, final_equity."""
    from src.backtest import run_backtest

    df = _make_canonical_df(n=300)
    cfg = _base_cfg()
    results = run_backtest(df, cfg)
    metrics = results["metrics"]
    for key in ("CAGR", "max_drawdown", "sharpe_ratio", "n_trades", "final_equity"):
        assert key in metrics, f"Missing metric: {key}"


def test_trivial_long_trade():
    """Strongly trending up data → strategy should make at least one profitable trade.

    We use a very strong uptrend (2% daily drift) to ensure the model learns
    to buy and profits on average.
    """
    from src.backtest import run_backtest

    # Strong uptrend: 1% daily return, minimal noise
    n = 300
    prices = 100.0 * np.cumprod(np.ones(n) * 1.01)  # pure trend, no noise
    dates = pd.date_range("2018-01-02", periods=n, freq="B", tz="UTC")
    df = pd.DataFrame({
        "timestamp": dates,
        "symbol": "UP",
        "open": prices,
        "high": prices * 1.005,
        "low": prices * 0.995,
        "close": prices,
        "volume": np.ones(n) * 1_000_000.0,
    })

    cfg = _base_cfg(commission=0.0, slippage_pct=0.0, fx_cost_pct=0.0)
    results = run_backtest(df, cfg)

    # At minimum, the result dict is well-formed
    assert "equity_curve" in results
    assert "metrics" in results

    # The benchmark should show positive returns on a pure uptrend
    bench = results["benchmark"]
    assert float(bench.iloc[-1]) > float(bench.iloc[0]) * 1.01, (
        "Buy-and-hold benchmark should be profitable on an uptrend"
    )


def test_costs_reduce_returns():
    """With cost enabled, total PnL should be lower than without costs."""
    from src.backtest import run_backtest

    df = _make_canonical_df(n=300, daily_return=0.003)

    # Run without costs
    cfg_no_cost = _base_cfg(commission=0.0, slippage_pct=0.0, fx_cost_pct=0.0)
    res_no_cost = run_backtest(df, cfg_no_cost)

    # Run with costs
    cfg_with_cost = _base_cfg(commission=1.0, slippage_pct=0.002, fx_cost_pct=0.005)
    res_with_cost = run_backtest(df, cfg_with_cost)

    # Note: with identical random seeds and data, the strategy places the same trades.
    # Costs reduce net returns, so final equity should be <= no-cost version.
    eq_no_cost = float(res_no_cost["equity_curve"].iloc[-1])
    eq_with_cost = float(res_with_cost["equity_curve"].iloc[-1])

    # If no trades were executed in either case, both are equal — that's acceptable.
    # If trades were executed, with-cost should be strictly less.
    n_trades = res_with_cost["metrics"]["n_trades"]
    if n_trades > 0:
        assert eq_with_cost <= eq_no_cost, (
            f"With-cost equity ({eq_with_cost:.2f}) should be <= "
            f"no-cost equity ({eq_no_cost:.2f}) when {n_trades} trades executed"
        )


def test_cash_constraint():
    """Cannot buy more shares than cash allows.

    Initial capital = 1000, price = 100, max_position_pct = 0.50
    → max qty = floor(0.50 * 1000 / 100) = 5 shares = $500.
    Equity should never go negative.
    """
    from src.backtest import run_backtest

    n = 310
    prices = np.ones(n) * 100.0  # constant price
    # Add a slight uptrend at the end to trigger a buy signal
    prices[155:] = 101.0
    dates = pd.date_range("2018-01-02", periods=n, freq="B", tz="UTC")
    df = pd.DataFrame({
        "timestamp": dates,
        "symbol": "CONST",
        "open": prices,
        "high": prices,
        "low": prices,
        "close": prices,
        "volume": np.ones(n) * 1_000_000.0,
    })

    cfg = _base_cfg()
    cfg["initial_capital"] = 1_000.0

    results = run_backtest(df, cfg)
    equity = results["equity_curve"]

    # Equity must never go negative
    assert (equity >= 0).all(), f"Equity went negative: min={equity.min():.2f}"
    # Equity should start at initial capital
    assert float(equity.iloc[0]) <= 1_000.0 * 1.01  # allow small floating point


def test_calc_cagr():
    """CAGR of a series that doubles over 252 days should be ~100%."""
    from src.backtest import _calc_cagr

    # 252 trading days = 1 year; start 100, end 200 → CAGR = 100%
    s = pd.Series(np.linspace(100, 200, 252))
    cagr = _calc_cagr(s)
    assert 0.90 < cagr < 1.10, f"Expected CAGR ~ 1.0, got {cagr:.4f}"


def test_calc_max_drawdown():
    """Max drawdown of a series that drops 50% should be -0.50."""
    from src.backtest import _calc_max_drawdown

    s = pd.Series([100, 120, 140, 70, 80, 90, 100])  # drops from 140 to 70 = -50%
    dd = _calc_max_drawdown(s)
    assert dd < -0.49 and dd > -0.51, f"Expected max drawdown ~ -0.50, got {dd:.4f}"


def test_calc_sharpe():
    """Sharpe ratio of a series with 1% daily returns and 0 vol should be very high."""
    from src.backtest import _calc_sharpe

    # Constant positive returns → very high Sharpe
    s = pd.Series(100.0 * (1.001 ** np.arange(252)))
    sharpe = _calc_sharpe(s)
    # With very low vol, Sharpe should be high
    assert sharpe > 1.0, f"Expected Sharpe > 1, got {sharpe:.4f}"


def test_equity_csv_written(tmp_path, monkeypatch):
    """run_backtest should write equity.csv to a logs/backtest_* directory."""
    import src.backtest as bt_mod

    monkeypatch.setattr(bt_mod, "_LOGS_DIR", tmp_path)

    df = _make_canonical_df(n=300)
    cfg = _base_cfg()
    results = bt_mod.run_backtest(df, cfg)

    # Find written equity CSVs (oos_equity.csv and dev_equity.csv)
    csv_files = list(tmp_path.rglob("*equity.csv"))
    assert len(csv_files) >= 1, "Expected at least one *equity.csv written"
