"""Regime-first trading strategy.

Two-layer decision:
  1. REGIME (rule-based, always checked first)
       Golden cross  (SMA50 > SMA200) → allowed to be long
       Death cross   (SMA50 < SMA200) → force exit regardless of ML
  2. ML TIMING (within regime)
       p > BUY_THRESHOLD  + golden cross → enter
       p < SELL_THRESHOLD OR death cross → exit

Philosophy: be in the market during confirmed uptrends (golden cross),
exit on trend reversals (death cross) or when ML turns clearly bearish.
Expected trade frequency: 15-30 per year.
"""
import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

BUY_THRESHOLD  = 0.48   # enter when golden cross + ML not bearish
SELL_THRESHOLD = 0.44   # exit when ML clearly bearish


def _predict(model, features: pd.DataFrame) -> float:
    """Unified predict_proba for LightGBM and LSTM models.

    Passes full feature history so LSTM can build sequences from tail.
    Returns probability of next period being up (float in [0, 1]).
    """
    if hasattr(model, "net"):
        from src.model_lstm import predict_proba
    else:
        from src.model import predict_proba

    proba = predict_proba(model, features)
    return float(proba[-1])


def decide(
    latest_features: pd.DataFrame,
    current_qty: int,
    model: object,
    cfg: dict[str, Any],
) -> tuple[str, int]:
    """Determine trading action from regime rules + ML timing.

    Returns:
        (action, qty) where action ∈ {'buy', 'sell', 'hold'} and qty >= 0.
    """
    if latest_features is None or latest_features.empty:
        logger.warning("[strategy] No features — holding.")
        return "hold", 0

    try:
        p = _predict(model, latest_features)
    except Exception as exc:
        logger.error("[strategy] predict failed: %s — holding.", exc)
        return "hold", 0

    X = latest_features.tail(1)

    # Regime signals from features
    sma50_vs_sma200 = 0.0
    if "SMA50_VS_SMA200" in X.columns:
        sma50_vs_sma200 = float(X["SMA50_VS_SMA200"].iloc[0])

    in_golden_cross = sma50_vs_sma200 > 0.0
    in_death_cross  = sma50_vs_sma200 < -0.005

    logger.info(
        "[strategy] p=%.4f golden=%s death=%s sma50_vs_200=%.4f qty=%d",
        p, in_golden_cross, in_death_cross, sma50_vs_sma200, current_qty,
    )

    # ── EXIT (checked before entry) ──────────────────────────────────────
    if current_qty > 0:
        if in_death_cross:
            logger.info("[strategy] SELL: death cross (SMA50/200=%.4f)", sma50_vs_sma200)
            return "sell", current_qty
        if p < SELL_THRESHOLD:
            logger.info("[strategy] SELL: ML bearish p=%.4f", p)
            return "sell", current_qty

    # ── ENTRY ─────────────────────────────────────────────────────────────
    if current_qty == 0 and in_golden_cross and p > BUY_THRESHOLD:
        logger.info("[strategy] BUY: golden cross + p=%.4f > %.2f", p, BUY_THRESHOLD)
        return "buy", 1

    return "hold", 0
