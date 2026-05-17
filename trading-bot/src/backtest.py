"""Walk-forward backtest with full cost modelling.

Strategy: train a fresh LightGBM model on each fold's training window,
predict on test window, execute simulated trades with realistic cost model.

Cost model per trade (buy and sell):
  - commission: flat fee (e.g. 0.0)
  - slippage:   price * slippage_pct per share
  - FX cost:    price * fx_cost_pct per share (once per trade direction)

Position sizing: floor(max_position_pct * equity / price), min 1 share.
Cash constraint: cannot buy if total cost exceeds available cash.
"""
import logging
from datetime import datetime, timezone
from math import floor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
_LOGS_DIR = _ROOT / "logs"


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _calc_cagr(equity_series: pd.Series) -> float:
    """Compound Annual Growth Rate."""
    if len(equity_series) < 2:
        return 0.0
    start_val = equity_series.iloc[0]
    end_val = equity_series.iloc[-1]
    if start_val <= 0:
        return 0.0
    # Number of years
    n_days = len(equity_series)
    years = n_days / 252.0
    if years <= 0:
        return 0.0
    return float((end_val / start_val) ** (1.0 / years) - 1.0)


def _calc_max_drawdown(equity_series: pd.Series) -> float:
    """Maximum drawdown (negative value, e.g. -0.25 for 25% drawdown)."""
    if len(equity_series) < 2:
        return 0.0
    roll_max = equity_series.cummax()
    drawdowns = (equity_series - roll_max) / roll_max
    return float(drawdowns.min())


def _calc_sharpe(equity_series: pd.Series) -> float:
    """Annualised Sharpe ratio (assuming 252 trading days, risk-free = 0)."""
    if len(equity_series) < 2:
        return 0.0
    daily_returns = equity_series.pct_change().dropna()
    if daily_returns.std() == 0:
        return 0.0
    return float(daily_returns.mean() / daily_returns.std() * np.sqrt(252))


# ---------------------------------------------------------------------------
# Core backtest logic
# ---------------------------------------------------------------------------

def _trade_cost(price: float, qty: int, commission: float, slippage_pct: float, fx_cost_pct: float) -> float:
    """Total transaction cost for a single trade."""
    slippage = price * slippage_pct * qty
    fx_cost = price * fx_cost_pct * qty
    return commission + slippage + fx_cost


def _run_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: dict[str, Any],
    initial_cash: float,
) -> tuple[list[dict], pd.Series]:
    """Run one walk-forward fold. Returns (trades, equity_series)."""
    from src.features import build_features
    from src.model import train as model_train, predict_proba

    # Build features for this fold
    try:
        X_train, y_train = build_features(train_df)
    except Exception as exc:
        logger.warning("[backtest] Fold train feature build failed: %s", exc)
        # Return flat equity
        dates = _extract_dates(test_df)
        return [], pd.Series(initial_cash, index=dates)

    if y_train is None or len(X_train) < 20:
        logger.warning("[backtest] Fold: not enough training data (%d rows)", len(X_train))
        dates = _extract_dates(test_df)
        return [], pd.Series(initial_cash, index=dates)

    # Train model
    try:
        model = model_train(X_train, y_train, cfg)
    except Exception as exc:
        logger.warning("[backtest] Fold model training failed: %s", exc)
        dates = _extract_dates(test_df)
        return [], pd.Series(initial_cash, index=dates)

    # Extract cost parameters
    bcfg = cfg.get("backtest", {})
    commission = float(bcfg.get("commission", 0.0))
    slippage_pct = float(bcfg.get("slippage_pct", 0.0002))
    fx_cost_pct = float(bcfg.get("fx_cost_pct", 0.005))
    max_pos_pct = float(cfg.get("risk", {}).get("max_position_pct", 0.10))
    buy_threshold = 0.55
    sell_threshold = 0.45

    cash = initial_cash
    # positions: symbol -> {qty, avg_price}
    positions: dict[str, dict] = {}
    trades: list[dict] = []

    # Sort test data by date, iterate day by day
    test_df = test_df.copy()
    test_df["timestamp"] = pd.to_datetime(test_df["timestamp"], utc=True)

    # Get unique dates in test window
    test_dates = sorted(test_df["timestamp"].dt.normalize().unique())
    equity_map: dict = {}

    # We need enough look-back rows for features; for each test day, we combine
    # the end of the train window with test rows so far
    cumulative_df = train_df.copy()

    for test_date in test_dates:
        # Rows for this date
        date_rows = test_df[test_df["timestamp"].dt.normalize() == test_date]

        if date_rows.empty:
            # Carry forward equity
            equity = _compute_equity(cash, positions, _last_prices(date_rows, test_df))
            equity_map[test_date] = equity
            continue

        # Append today's rows to cumulative for feature building
        cumulative_df = pd.concat([cumulative_df, date_rows], ignore_index=True)

        symbols_today = date_rows["symbol"].unique()

        for sym in symbols_today:
            sym_rows = date_rows[date_rows["symbol"] == sym]
            if sym_rows.empty:
                continue

            # Current bar's close price
            current_price = float(sym_rows["close"].iloc[-1])
            if current_price <= 0:
                continue

            # Build features from cumulative data for this symbol
            sym_cumulative = cumulative_df[cumulative_df["symbol"] == sym].copy()

            try:
                X_feat, _ = build_features(sym_cumulative)
            except Exception:
                continue

            if X_feat.empty:
                continue

            # Use last row for inference
            X_last = X_feat.tail(1)
            proba = predict_proba(model, X_last)[0]

            current_qty = positions.get(sym, {}).get("qty", 0)
            desired_side = None
            desired_qty = 0

            if proba > buy_threshold and current_qty == 0:
                desired_side = "buy"
                desired_qty = max(1, floor(max_pos_pct * cash / current_price))
            elif proba < sell_threshold and current_qty > 0:
                desired_side = "sell"
                desired_qty = current_qty

            if desired_side is None or desired_qty < 1:
                continue

            if desired_side == "buy":
                cost = current_price * desired_qty + _trade_cost(
                    current_price, desired_qty, commission, slippage_pct, fx_cost_pct
                )
                if cost > cash:
                    # Reduce qty to what cash allows
                    # (cost per share = price + slippage_pct*price + fx_cost_pct*price)
                    cost_per_share = current_price * (1 + slippage_pct + fx_cost_pct)
                    desired_qty = max(0, floor((cash - commission) / cost_per_share))
                    if desired_qty < 1:
                        continue
                    cost = current_price * desired_qty + _trade_cost(
                        current_price, desired_qty, commission, slippage_pct, fx_cost_pct
                    )

                cash -= cost
                avg_price = current_price
                entry_cost = _trade_cost(current_price, desired_qty, commission, slippage_pct, fx_cost_pct)
                positions[sym] = {"qty": desired_qty, "avg_price": avg_price}
                trades.append({
                    "date": test_date,
                    "symbol": sym,
                    "side": "buy",
                    "qty": desired_qty,
                    "price": current_price,
                    "cost": entry_cost,
                    "pnl": 0.0,
                })

            elif desired_side == "sell" and sym in positions:
                qty = positions[sym]["qty"]
                avg_entry = positions[sym]["avg_price"]
                exit_cost = _trade_cost(current_price, qty, commission, slippage_pct, fx_cost_pct)
                gross_proceeds = current_price * qty
                net_proceeds = gross_proceeds - exit_cost
                pnl = (current_price - avg_entry) * qty - exit_cost
                # Also subtract entry cost from PnL (already paid at buy time)
                cash += net_proceeds
                del positions[sym]
                trades.append({
                    "date": test_date,
                    "symbol": sym,
                    "side": "sell",
                    "qty": qty,
                    "price": current_price,
                    "cost": exit_cost,
                    "pnl": pnl,
                })

        # Compute equity for this date
        price_map = {sym: float(test_df[test_df["symbol"] == sym]["close"].iloc[-1])
                     for sym in test_df["symbol"].unique()}
        equity = cash + sum(
            pos["qty"] * price_map.get(sym, pos["avg_price"])
            for sym, pos in positions.items()
        )
        equity_map[test_date] = equity

    if not equity_map:
        dates = _extract_dates(test_df)
        return trades, pd.Series(initial_cash, index=dates)

    equity_series = pd.Series(equity_map).sort_index()
    return trades, equity_series


def _extract_dates(df: pd.DataFrame) -> pd.DatetimeIndex:
    ts = pd.to_datetime(df["timestamp"], utc=True)
    return pd.DatetimeIndex(sorted(ts.dt.normalize().unique()))


def _last_prices(date_rows: pd.DataFrame, full_df: pd.DataFrame) -> dict:
    prices = {}
    for sym in full_df["symbol"].unique():
        sym_rows = full_df[full_df["symbol"] == sym]
        if not sym_rows.empty:
            prices[sym] = float(sym_rows["close"].iloc[-1])
    return prices


def _compute_equity(cash: float, positions: dict, price_map: dict) -> float:
    equity = cash
    for sym, pos in positions.items():
        price = price_map.get(sym, pos["avg_price"])
        equity += pos["qty"] * price
    return equity


def run_backtest(df: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    """Walk-forward backtest with N folds.

    Args:
        df:  canonical DataFrame (timestamp, symbol, open, high, low, close, volume)
        cfg: config dict (from load_config())

    Returns dict with:
        equity_curve:  pd.Series of daily equity indexed by date
        trades:        list of trade dicts
        metrics:       dict {CAGR, max_drawdown, sharpe_ratio, n_trades, final_equity}
        benchmark:     buy-and-hold equity curve
    Also writes equity CSV and metrics text to logs/backtest_<timestamp>/.
    """
    n_folds = 5
    initial_capital = float(cfg.get("initial_capital", 10_000))

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    # Get unique dates across all symbols
    all_dates = sorted(df["timestamp"].dt.normalize().unique())
    n_dates = len(all_dates)

    if n_dates < n_folds * 2:
        raise ValueError(
            f"[backtest] Not enough dates ({n_dates}) to create {n_folds} folds."
        )

    fold_size = n_dates // n_folds
    # Minimum training window: 60% of fold size or at least 60 days
    min_train = max(60, int(fold_size * 0.6))

    all_trades: list[dict] = []
    equity_segments: list[pd.Series] = []
    current_equity = initial_capital

    logger.info(
        "[backtest] Walk-forward: %d dates, %d folds, fold_size=%d",
        n_dates,
        n_folds,
        fold_size,
    )

    for fold_idx in range(n_folds):
        test_start_idx = fold_idx * fold_size
        test_end_idx = test_start_idx + fold_size if fold_idx < n_folds - 1 else n_dates

        # Train on all data before this fold's test window
        train_end_idx = test_start_idx

        if train_end_idx < min_train:
            logger.warning("[backtest] Fold %d: not enough training data, skipping.", fold_idx)
            # Carry forward equity
            test_dates = all_dates[test_start_idx:test_end_idx]
            eq_seg = pd.Series(current_equity, index=pd.DatetimeIndex(test_dates))
            equity_segments.append(eq_seg)
            continue

        train_dates = all_dates[:train_end_idx]
        test_dates = all_dates[test_start_idx:test_end_idx]

        train_mask = df["timestamp"].dt.normalize().isin(train_dates)
        test_mask = df["timestamp"].dt.normalize().isin(test_dates)

        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()

        logger.info(
            "[backtest] Fold %d: train=%d rows, test=%d rows, starting equity=%.2f",
            fold_idx,
            len(train_df),
            len(test_df),
            current_equity,
        )

        fold_trades, fold_equity = _run_fold(train_df, test_df, cfg, current_equity)

        all_trades.extend(fold_trades)
        equity_segments.append(fold_equity)

        if not fold_equity.empty:
            current_equity = float(fold_equity.iloc[-1])

    # Combine equity segments
    if equity_segments:
        equity_curve = pd.concat(equity_segments).sort_index()
        # Prepend starting equity
        start_date = pd.Timestamp(all_dates[0]) - pd.Timedelta(days=1)
        start_point = pd.Series({start_date: initial_capital})
        equity_curve = pd.concat([start_point, equity_curve]).sort_index()
    else:
        equity_curve = pd.Series({pd.Timestamp(all_dates[0]): initial_capital})

    # Buy-and-hold benchmark
    # Use first symbol, buy at start, sell at end
    symbols = df["symbol"].unique()
    bench_sym = symbols[0]
    sym_df = df[df["symbol"] == bench_sym].sort_values("timestamp")

    bench_dates = sorted(sym_df["timestamp"].dt.normalize().unique())
    bench_prices_map = (
        sym_df.groupby(sym_df["timestamp"].dt.normalize())["close"].last()
    )
    bench_start_price = float(bench_prices_map.iloc[0])
    bench_shares = floor(initial_capital / bench_start_price)
    bench_cash = initial_capital - bench_shares * bench_start_price

    benchmark_equity = pd.Series(
        {
            date: bench_cash + bench_shares * float(bench_prices_map.get(date, bench_start_price))
            for date in bench_prices_map.index
        }
    ).sort_index()

    # Metrics
    cagr = _calc_cagr(equity_curve)
    max_dd = _calc_max_drawdown(equity_curve)
    sharpe = _calc_sharpe(equity_curve)
    final_equity = float(equity_curve.iloc[-1])
    n_trades = len(all_trades)

    metrics = {
        "CAGR": cagr,
        "max_drawdown": max_dd,
        "sharpe_ratio": sharpe,
        "n_trades": n_trades,
        "final_equity": final_equity,
        "initial_equity": initial_capital,
        "total_return": (final_equity - initial_capital) / initial_capital,
    }

    # Benchmark metrics
    bm_cagr = _calc_cagr(benchmark_equity)
    bm_max_dd = _calc_max_drawdown(benchmark_equity)
    bm_sharpe = _calc_sharpe(benchmark_equity)
    bm_final = float(benchmark_equity.iloc[-1])

    # Save outputs
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = _LOGS_DIR / f"backtest_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    equity_csv_path = out_dir / "equity.csv"
    equity_curve.to_csv(equity_csv_path, header=["equity"])
    logger.info("[backtest] Equity curve saved to %s", equity_csv_path)

    metrics_txt_path = out_dir / "metrics.txt"
    with open(metrics_txt_path, "w") as f:
        f.write("=== Strategy Metrics ===\n")
        for k, v in metrics.items():
            f.write(f"  {k}: {v:.4f}\n" if isinstance(v, float) else f"  {k}: {v}\n")
        f.write("\n=== Buy-and-Hold Benchmark ===\n")
        f.write(f"  CAGR: {bm_cagr:.4f}\n")
        f.write(f"  max_drawdown: {bm_max_dd:.4f}\n")
        f.write(f"  sharpe_ratio: {bm_sharpe:.4f}\n")
        f.write(f"  final_equity: {bm_final:.2f}\n")
    logger.info("[backtest] Metrics saved to %s", metrics_txt_path)

    # Print comparison table
    print("\n" + "=" * 60)
    print(f"{'Metric':<25} {'Strategy':>12} {'Buy&Hold':>12}")
    print("-" * 60)
    print(f"{'CAGR':<25} {cagr:>12.2%} {bm_cagr:>12.2%}")
    print(f"{'Max Drawdown':<25} {max_dd:>12.2%} {bm_max_dd:>12.2%}")
    print(f"{'Sharpe Ratio':<25} {sharpe:>12.3f} {bm_sharpe:>12.3f}")
    print(f"{'Final Equity':<25} {final_equity:>12.2f} {bm_final:>12.2f}")
    print(f"{'N Trades':<25} {n_trades:>12}")
    print("=" * 60 + "\n")

    return {
        "equity_curve": equity_curve,
        "trades": all_trades,
        "metrics": metrics,
        "benchmark": benchmark_equity,
    }
