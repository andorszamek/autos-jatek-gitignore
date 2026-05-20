"""Model signal -> target position.

decide() converts a model probability + trend filter into a trading action.
The risk module finalises the actual order quantity.

Thresholds:
  BUY_THRESHOLD  = 0.52  (was 0.55 — trades much more frequently)
  SELL_THRESHOLD = 0.48  (was 0.45)

Trend filter (PRICE_VS_SMA200):
  Only enter long when price is above the 200-day SMA (bull regime).
  Will still exit (sell) regardless of trend when sell signal fires.
"""
import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

BUY_THRESHOLD  = 0.505
SELL_THRESHOLD = 0.495


def decide(
    latest_features: pd.DataFrame,
    current_qty: int,
    model: object,
    cfg: dict[str, Any],
) -> tuple[str, int]:
    """Determine trading action from model probability + trend filter.

    Returns:
        (action, qty) where action ∈ {'buy', 'sell', 'hold'} and qty >= 0.
        qty=1 for buy (RiskManager finalises actual shares), current_qty for sell.
    """
    from src.model import predict_proba

    if latest_features is None or latest_features.empty:
        logger.warning("[strategy] No features — holding.")
        return "hold", 0

    X = latest_features.tail(1)

    try:
        proba = predict_proba(model, X)
        p = float(proba[0])
    except Exception as exc:
        logger.error("[strategy] predict_proba failed: %s — holding.", exc)
        return "hold", 0

    # Trend filter: only go long when price is above the 200-day SMA.
    in_bull_regime = True
    if "PRICE_VS_SMA200" in X.columns:
        price_vs_sma200 = float(X["PRICE_VS_SMA200"].iloc[0])
        in_bull_regime = price_vs_sma200 > -0.02  # allow up to 2% below SMA200
        if not in_bull_regime:
            logger.info(
                "[strategy] Bear regime (price %.1f%% below SMA200) — no new buys.",
                price_vs_sma200 * 100,
            )

    logger.info("[strategy] p=%.4f in_bull=%s current_qty=%d", p, in_bull_regime, current_qty)

    if p > BUY_THRESHOLD and current_qty == 0 and in_bull_regime:
        logger.info("[strategy] BUY signal: p=%.4f", p)
        return "buy", 1  # RiskManager sets actual qty

    if p < SELL_THRESHOLD and current_qty > 0:
        logger.info("[strategy] SELL signal: p=%.4f qty=%d", p, current_qty)
        return "sell", current_qty

    return "hold", 0
