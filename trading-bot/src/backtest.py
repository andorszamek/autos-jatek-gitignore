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
    pretrained_model=None,
) -> tuple[list[dict], pd.Series]:
    """Run one walk-forward fold. Returns (trades, equity_series)."""
    from src.features import build_features

    model_type = cfg.get("_model_type", "lgbm")
    if model_type == "lstm":
        from src.model_lstm import train as model_train, predict_proba
    else:
        from src.model import train as model_train, predict_proba

    if pretrained_model is not None:
        model = pretrained_model
    else:
        # Build features and train fresh model for this fold
        try:
            X_train, y_train = build_features(train_df)
        except Exception as exc:
            logger.warning("[backtest] Fold train feature build failed: %s", exc)
            dates = _extract_dates(test_df)
            return [], pd.Series(initial_cash, index=dates)

        if y_train is None or len(X_train) < 20:
            logger.warning("[backtest] Fold: not enough training data (%d rows)", len(X_train))
            dates = _extract_dates(test_df)
            return [], pd.Series(initial_cash, index=dates)

        try:
            model = model_train(X_train, y_train, cfg)
        except Exception as exc:
            logger.warning("[backtest] Fold model training failed: %s", exc)
            dates = _extract_dates(test_df)
            return [], pd.Series(initial_cash, index=dates)

    # Extract cost parameters
    from src.strategy import BUY_THRESHOLD, SELL_THRESHOLD, momentum_rank

    bcfg = cfg.get("backtest", {})
    commission = float(bcfg.get("commission", 0.0))
    slippage_pct = float(bcfg.get("slippage_pct", 0.0002))
    fx_cost_pct = float(bcfg.get("fx_cost_pct", 0.005))
    stop_loss_pct = float(cfg.get("risk", {}).get("stop_loss_pct", 0.03))
    universe = cfg.get("universe", sorted(train_df["symbol"].unique().tolist()))
    buy_threshold = BUY_THRESHOLD
    sell_threshold = SELL_THRESHOLD

    cash = initial_cash
    # positions: symbol -> {qty, avg_price}  (at most 1 position held at a time)
    positions: dict[str, dict] = {}
    trades: list[dict] = []

    # Sort test data by date, iterate day by day
    test_df = test_df.copy()
    test_df["timestamp"] = pd.to_datetime(test_df["timestamp"], utc=True)

    # Get unique dates in test window
    test_dates = sorted(test_df["timestamp"].dt.normalize().unique())
    equity_map: dict = {}

    # Cumulative look-back buffer for feature building
    cumulative_df = train_df.copy()

    for test_date in test_dates:
        date_rows = test_df[test_df["timestamp"].dt.normalize() == test_date]

        if date_rows.empty:
            equity_map[test_date] = _compute_equity(cash, positions, {})
            continue

        # Grow look-back buffer, trim to last 280 rows per symbol to cap memory
        cumulative_df = pd.concat([cumulative_df, date_rows], ignore_index=True)
        if len(cumulative_df) > 300:
            cumulative_df = (
                cumulative_df
                .groupby("symbol", group_keys=False)
                .tail(280)
                .reset_index(drop=True)
            )

        # ── Build features for every symbol present today ─────────────────
        features_by_sym: dict[str, pd.DataFrame] = {}
        price_by_sym: dict[str, float] = {}

        for sym in universe:
            sym_rows = date_rows[date_rows["symbol"] == sym]
            if sym_rows.empty:
                continue
            p = float(sym_rows["close"].iloc[-1])
            if p <= 0:
                continue
            price_by_sym[sym] = p
            sym_cumulative = cumulative_df[cumulative_df["symbol"] == sym].copy()
            try:
                X_feat, _ = build_features(sym_cumulative)
            except Exception:
                continue
            if not X_feat.empty:
                features_by_sym[sym] = X_feat

        if not features_by_sym:
            equity_map[test_date] = _compute_equity(cash, positions, price_by_sym)
            continue

        # ── Dual-momentum rotation: pick best asset ────────────────────────
        target_sym = momentum_rank(features_by_sym)

        # ── Sell any position that is NOT the target (rotation exit) ───────
        for held_sym in list(positions.keys()):
            if held_sym == target_sym:
                continue
            p = price_by_sym.get(held_sym, positions[held_sym]["avg_price"])
            qty = positions[held_sym]["qty"]
            avg_e = positions[held_sym]["avg_price"]
            exit_cost = _trade_cost(p, qty, commission, slippage_pct, fx_cost_pct)
            pnl = (p - avg_e) * qty - exit_cost
            cash += p * qty - exit_cost
            del positions[held_sym]
            trades.append({
                "date": test_date,
                "symbol": held_sym,
                "side": "sell",
                "qty": qty,
                "price": p,
                "cost": exit_cost,
                "pnl": pnl,
                "reason": "rotation",
            })

        # ── Manage the target symbol position ─────────────────────────────
        if target_sym and target_sym in features_by_sym:
            current_price = price_by_sym[target_sym]
            X_target = features_by_sym[target_sym]
            current_qty = positions.get(target_sym, {}).get("qty", 0)

            # ML probability for target symbol
            if model_type == "lstm":
                proba_arr = predict_proba(model, X_target)
                proba = float(proba_arr[-1]) if len(proba_arr) > 0 else 0.5
            else:
                proba = float(predict_proba(model, X_target.tail(1))[0])

            # Regime check
            sma50_vs_sma200 = 0.0
            if "SMA50_VS_SMA200" in X_target.columns:
                sma50_vs_sma200 = float(X_target["SMA50_VS_SMA200"].iloc[-1])
            in_golden_cross = sma50_vs_sma200 > 0.0
            in_death_cross  = sma50_vs_sma200 < -0.005

            desired_side: str | None = None
            desired_qty = 0

            # Stop-loss (highest priority)
            if current_qty > 0:
                entry_price = positions[target_sym]["avg_price"]
                if current_price < entry_price * (1 - stop_loss_pct):
                    desired_side = "sell"
                    desired_qty = current_qty

            # ML + regime signals
            if desired_side is None:
                if current_qty > 0 and (in_death_cross or proba < sell_threshold):
                    desired_side = "sell"
                    desired_qty = current_qty
                elif current_qty == 0 and in_golden_cross and proba > buy_threshold:
                    desired_side = "buy"
                    desired_qty = max(1, floor(cash / current_price * 0.99))  # use ~all cash

            if desired_side == "buy" and desired_qty >= 1:
                cost = current_price * desired_qty + _trade_cost(
                    current_price, desired_qty, commission, slippage_pct, fx_cost_pct
                )
                if cost > cash:
                    cost_per_share = current_price * (1 + slippage_pct + fx_cost_pct)
                    desired_qty = max(0, floor((cash - commission) / cost_per_share))
                    if desired_qty < 1:
                        desired_side = None
                    else:
                        cost = current_price * desired_qty + _trade_cost(
                            current_price, desired_qty, commission, slippage_pct, fx_cost_pct
                        )
                if desired_side and desired_qty >= 1:
                    entry_cost = _trade_cost(current_price, desired_qty, commission, slippage_pct, fx_cost_pct)
                    cash -= cost
                    positions[target_sym] = {"qty": desired_qty, "avg_price": current_price}
                    trades.append({
                        "date": test_date,
                        "symbol": target_sym,
                        "side": "buy",
                        "qty": desired_qty,
                        "price": current_price,
                        "cost": entry_cost,
                        "pnl": 0.0,
                        "reason": "entry",
                    })

            elif desired_side == "sell" and target_sym in positions:
                qty = positions[target_sym]["qty"]
                avg_entry = positions[target_sym]["avg_price"]
                exit_cost = _trade_cost(current_price, qty, commission, slippage_pct, fx_cost_pct)
                pnl = (current_price - avg_entry) * qty - exit_cost
                cash += current_price * qty - exit_cost
                del positions[target_sym]
                trades.append({
                    "date": test_date,
                    "symbol": target_sym,
                    "side": "sell",
                    "qty": qty,
                    "price": current_price,
                    "cost": exit_cost,
                    "pnl": pnl,
                    "reason": "signal",
                })

        # ── Mark-to-market equity ─────────────────────────────────────────
        equity_map[test_date] = _compute_equity(cash, positions, price_by_sym)

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
    n_folds = 4
    initial_capital = float(cfg.get("initial_capital", 10_000))
    oos_fraction = float(cfg.get("backtest", {}).get("oos_fraction", 0.20))

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    all_dates = sorted(df["timestamp"].dt.normalize().unique())
    n_dates = len(all_dates)

    if n_dates < 300:
        raise ValueError(f"[backtest] Not enough dates ({n_dates}), need at least 300.")

    # Split: development (first 80%) + true OOS hold-out (last 20%)
    n_oos = max(50, int(n_dates * oos_fraction))
    dev_dates = all_dates[:-n_oos]
    oos_dates = all_dates[-n_oos:]
    n_dev = len(dev_dates)

    logger.info(
        "[backtest] Dev: %d dates, OOS hold-out: %d dates (%.0f%%)",
        n_dev, n_oos, oos_fraction * 100,
    )

    # Load pre-trained model if --use-saved (skips per-fold training)
    pretrained = None
    if cfg.get("_use_saved_model"):
        model_type = cfg.get("_model_type", "lgbm")
        model_path = str(_ROOT / "models" / ("latest.lstm" if model_type == "lstm" else "latest.lgb"))
        if model_type == "lstm":
            from src.model_lstm import load as _load_model
        else:
            from src.model import load as _load_model
        pretrained = _load_model(model_path)
        print(f"[backtest] Using pre-trained model: {model_path}")

    # ---- Walk-forward on development data ----
    fold_size = n_dev // n_folds
    min_train = max(60, int(fold_size * 0.6))

    all_trades: list[dict] = []
    equity_segments: list[pd.Series] = []
    current_equity = initial_capital

    for fold_idx in range(n_folds):
        test_start_idx = fold_idx * fold_size
        test_end_idx = test_start_idx + fold_size if fold_idx < n_folds - 1 else n_dev

        train_end_idx = test_start_idx
        if train_end_idx < min_train:
            logger.warning("[backtest] Fold %d: not enough training data, skipping.", fold_idx)
            test_dates_fold = dev_dates[test_start_idx:test_end_idx]
            equity_segments.append(pd.Series(current_equity, index=pd.DatetimeIndex(test_dates_fold)))
            continue

        train_dates_fold = dev_dates[:train_end_idx]
        test_dates_fold = dev_dates[test_start_idx:test_end_idx]

        train_df = df[df["timestamp"].dt.normalize().isin(set(train_dates_fold))].copy()
        test_df = df[df["timestamp"].dt.normalize().isin(set(test_dates_fold))].copy()

        logger.info(
            "[backtest] Fold %d: train=%d rows, test=%d rows",
            fold_idx, len(train_df), len(test_df),
        )

        fold_trades, fold_equity = _run_fold(train_df, test_df, cfg, current_equity, pretrained_model=pretrained)
        all_trades.extend(fold_trades)
        equity_segments.append(fold_equity)
        if not fold_equity.empty:
            current_equity = float(fold_equity.iloc[-1])

    if equity_segments:
        dev_equity = pd.concat(equity_segments).sort_index()
        start_date = pd.Timestamp(dev_dates[0]) - pd.Timedelta(days=1)
        dev_equity = pd.concat([pd.Series({start_date: initial_capital}), dev_equity]).sort_index()
    else:
        dev_equity = pd.Series({pd.Timestamp(dev_dates[0]): initial_capital})

    # ---- True OOS evaluation: train on ALL dev data, test on hold-out ----
    print("\n[backtest] Running true OOS evaluation on held-out period...")
    oos_train_df = df[df["timestamp"].dt.normalize().isin(set(dev_dates))].copy()
    oos_test_df = df[df["timestamp"].dt.normalize().isin(set(oos_dates))].copy()
    oos_trades, oos_equity_raw = _run_fold(oos_train_df, oos_test_df, cfg, initial_capital, pretrained_model=pretrained)

    oos_start = pd.Timestamp(oos_dates[0]) - pd.Timedelta(days=1)
    oos_equity = pd.concat([pd.Series({oos_start: initial_capital}), oos_equity_raw]).sort_index()

    # ---- Buy-and-hold benchmark (SPY, or first symbol if SPY not present) ----
    bench_sym = "SPY" if "SPY" in df["symbol"].unique() else df["symbol"].unique()[0]
    sym_df = df[df["symbol"] == bench_sym].sort_values("timestamp")
    bench_prices = sym_df.groupby(sym_df["timestamp"].dt.normalize())["close"].last()

    def _bh_equity(dates, cap):
        start_p = float(bench_prices[bench_prices.index.isin(dates)].iloc[0])
        shares = floor(cap / start_p)
        cash_rem = cap - shares * start_p
        return pd.Series(
            {d: cash_rem + shares * float(bench_prices.get(d, start_p))
             for d in bench_prices.index if d in set(dates)}
        ).sort_index()

    bh_full = _bh_equity(all_dates, initial_capital)
    bh_oos = _bh_equity(oos_dates, initial_capital)

    # ---- Metrics ----
    dev_cagr = _calc_cagr(dev_equity)
    dev_dd = _calc_max_drawdown(dev_equity)
    dev_sharpe = _calc_sharpe(dev_equity)

    oos_cagr = _calc_cagr(oos_equity)
    oos_dd = _calc_max_drawdown(oos_equity)
    oos_sharpe = _calc_sharpe(oos_equity)
    oos_n_trades = len(oos_trades)

    bh_full_cagr = _calc_cagr(bh_full)
    bh_oos_cagr = _calc_cagr(bh_oos)
    bh_oos_dd = _calc_max_drawdown(bh_oos)
    bh_oos_sharpe = _calc_sharpe(bh_oos)

    n_dev_trades = len(all_trades)

    metrics = {
        "CAGR": oos_cagr,
        "max_drawdown": oos_dd,
        "sharpe_ratio": oos_sharpe,
        "n_trades": oos_n_trades,
        "final_equity": float(oos_equity.iloc[-1]),
        "initial_equity": initial_capital,
        "total_return": (float(oos_equity.iloc[-1]) - initial_capital) / initial_capital,
    }

    # ---- Save outputs ----
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = _LOGS_DIR / f"backtest_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    oos_equity.to_csv(out_dir / "oos_equity.csv", header=["equity"])
    dev_equity.to_csv(out_dir / "dev_equity.csv", header=["equity"])

    with open(out_dir / "metrics.txt", "w") as f:
        f.write(f"Dev walk-forward CAGR : {dev_cagr:.4f}\n")
        f.write(f"OOS CAGR              : {oos_cagr:.4f}\n")
        f.write(f"OOS Max Drawdown      : {oos_dd:.4f}\n")
        f.write(f"OOS Sharpe            : {oos_sharpe:.4f}\n")
        f.write(f"OOS N Trades          : {oos_n_trades}\n")
        f.write(f"B&H full CAGR         : {bh_full_cagr:.4f}\n")
        f.write(f"B&H OOS CAGR          : {bh_oos_cagr:.4f}\n")

    # ---- Print ----
    print("\n" + "=" * 62)
    print(f"  DEV walk-forward ({n_dev} days, {n_dev_trades} trades)")
    print(f"  CAGR {dev_cagr:+.2%}  |  MaxDD {dev_dd:.2%}  |  Sharpe {dev_sharpe:.3f}")
    print("-" * 62)
    print(f"{'Metric':<25} {'OOS Strategy':>15} {'OOS Buy&Hold':>15}")
    print("-" * 62)
    print(f"{'CAGR':<25} {oos_cagr:>15.2%} {bh_oos_cagr:>15.2%}")
    print(f"{'Max Drawdown':<25} {oos_dd:>15.2%} {bh_oos_dd:>15.2%}")
    print(f"{'Sharpe Ratio':<25} {oos_sharpe:>15.3f} {bh_oos_sharpe:>15.3f}")
    print(f"{'Final Equity':<25} {float(oos_equity.iloc[-1]):>15.2f} {float(bh_oos.iloc[-1]):>15.2f}")
    print(f"{'N Trades (OOS)':<25} {oos_n_trades:>15}")
    print("=" * 62 + "\n")
    print(f"  Buy&Hold full period CAGR: {bh_full_cagr:.2%}\n")

    return {
        "equity_curve": oos_equity,
        "dev_equity": dev_equity,
        "trades": oos_trades,
        "metrics": metrics,
        "benchmark": bh_oos,
    }
