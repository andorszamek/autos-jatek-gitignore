"""Model signal -> target position.

decide() converts a model probability into a trading action and quantity.
The risk module finalises the actual order quantity.
"""
import logging
from math import floor
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

BUY_THRESHOLD = 0.55
SELL_THRESHOLD = 0.45


def decide(
    latest_features: pd.DataFrame,
    current_qty: int,
    model: object,
    cfg: dict[str, Any],
) -> tuple[str, int]:
    """Determine trading action from model probability.

    Args:
        latest_features:  DataFrame with FEATURE_COLUMNS (one or more rows; uses last row)
        current_qty:      current position in shares (int, 0 = flat)
        model:            trained lgb.Booster (or any object with predict())
        cfg:              config dict

    Returns:
        (action, qty) where action ∈ {'buy', 'sell', 'hold'} and qty >= 0
        qty is always 0 for 'hold', positive integer for 'buy'/'sell'.
    """
    from src.model import predict_proba

    if latest_features is None or latest_features.empty:
        logger.warning("[strategy] No features provided — holding.")
        return "hold", 0

    # Use last row for inference
    X = latest_features.tail(1)

    try:
        proba = predict_proba(model, X)
        p = float(proba[0])
    except Exception as exc:
        logger.error("[strategy] predict_proba failed: %s — holding.", exc)
        return "hold", 0

    logger.debug("[strategy] p=%.4f current_qty=%d", p, current_qty)

    # Signal logic
    if p > BUY_THRESHOLD and current_qty == 0:
        # Rough sizing: max_position_pct * equity, but we don't have price/equity here.
        # Return qty=1 as a placeholder; RiskManager will finalize.
        qty = 1
        logger.info("[strategy] BUY signal: p=%.4f > %.2f", p, BUY_THRESHOLD)
        return "buy", qty

    if p < SELL_THRESHOLD and current_qty > 0:
        logger.info("[strategy] SELL signal: p=%.4f < %.2f, qty=%d", p, SELL_THRESHOLD, current_qty)
        return "sell", current_qty

    logger.debug("[strategy] HOLD: p=%.4f", p)
    return "hold", 0
