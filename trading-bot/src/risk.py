"""Risk management — position sizing, stop-loss, daily limit, kill switch."""
import logging
from math import floor
from typing import Any

logger = logging.getLogger(__name__)


class RiskManager:
    """Stateful risk manager tracking peak equity for kill switch."""

    def __init__(self, cfg: dict, initial_equity: float) -> None:
        self._cfg = cfg
        self._peak_equity = initial_equity
        risk_cfg = cfg.get("risk", {})
        self._max_position_pct: float = float(risk_cfg.get("max_position_pct", 0.10))
        self._stop_loss_pct: float = float(risk_cfg.get("stop_loss_pct", 0.03))
        self._daily_max_loss_pct: float = float(risk_cfg.get("daily_max_loss_pct", 0.04))
        self._kill_switch_pct: float = float(risk_cfg.get("kill_switch_drawdown_pct", 0.20))

    def check_order(
        self,
        symbol: str,
        desired_qty: int,
        side: str,
        current_price: float,
        equity: float,
        daily_start_equity: float,
    ) -> tuple[bool, int, str]:
        """Validate and size an order.

        Returns:
            (approved: bool, approved_qty: int, reason: str)
        """
        # 1. Kill switch: drawdown from peak >= kill_switch_drawdown_pct
        if self._peak_equity > 0:
            drawdown = (self._peak_equity - equity) / self._peak_equity
            if drawdown >= self._kill_switch_pct:
                msg = (
                    f"KILL SWITCH: drawdown {drawdown:.2%} >= "
                    f"threshold {self._kill_switch_pct:.2%}"
                )
                logger.critical("[risk] %s | symbol=%s", msg, symbol)
                return False, 0, msg

        # 2. Daily loss limit
        if daily_start_equity > 0:
            daily_pnl_pct = (equity - daily_start_equity) / daily_start_equity
            if daily_pnl_pct <= -self._daily_max_loss_pct:
                msg = (
                    f"Daily loss limit hit: {daily_pnl_pct:.2%} <= "
                    f"-{self._daily_max_loss_pct:.2%}"
                )
                logger.warning("[risk] %s | symbol=%s", msg, symbol)
                return False, 0, msg

        # 3. Position size cap
        if current_price > 0 and equity > 0:
            max_qty = floor(self._max_position_pct * equity / current_price)
        else:
            max_qty = 0

        approved_qty = min(desired_qty, max_qty)

        # 4. Must be at least 1 share
        if approved_qty < 1:
            msg = (
                f"Rejected: approved_qty={approved_qty} < 1 "
                f"(max_position_pct={self._max_position_pct:.2%}, "
                f"equity={equity:.2f}, price={current_price:.2f})"
            )
            logger.info("[risk] %s | symbol=%s", msg, symbol)
            return False, 0, msg

        msg = (
            f"Approved: qty={approved_qty} (desired={desired_qty}, "
            f"max_allowed={max_qty})"
        )
        logger.debug("[risk] %s | symbol=%s side=%s", msg, symbol, side)
        return True, approved_qty, msg

    def check_stop_loss(
        self,
        symbol: str,
        current_price: float,
        entry_price: float,
    ) -> bool:
        """Return True if stop-loss is triggered (loss >= stop_loss_pct).

        Loss is calculated as: (entry_price - current_price) / entry_price
        Triggered when loss >= stop_loss_pct (e.g. 0.03 = 3%).
        """
        if entry_price <= 0:
            return False
        loss = (entry_price - current_price) / entry_price
        triggered = loss >= self._stop_loss_pct
        if triggered:
            logger.warning(
                "[risk] Stop-loss triggered for %s: entry=%.4f current=%.4f loss=%.4f >= threshold=%.4f",
                symbol,
                entry_price,
                current_price,
                loss,
                self._stop_loss_pct,
            )
        return triggered

    def update_peak(self, equity: float) -> None:
        """Update peak equity tracker. Call after each cycle."""
        if equity > self._peak_equity:
            self._peak_equity = equity
            logger.debug("[risk] Peak equity updated to %.2f", equity)


# ---------------------------------------------------------------------------
# Module-level backward-compatible function (stub signature kept)
# ---------------------------------------------------------------------------

def check_order(
    symbol: str,
    desired_qty: float,
    side: str,
    equity: float,
    daily_pnl: float,
    peak_equity: float,
    cfg: dict[str, Any],
) -> tuple[bool, float, str]:
    """Module-level convenience wrapper (backward compatibility).

    Args:
        symbol:       trading symbol
        desired_qty:  number of shares requested (will be cast to int)
        side:         'buy' or 'sell'
        equity:       current total equity
        daily_pnl:    unrealised+realised PnL for today (negative = loss)
        peak_equity:  highest equity ever reached (for kill switch)
        cfg:          config dict

    Returns:
        (approved: bool, approved_qty: float, reason: str)
    """
    risk_cfg = cfg.get("risk", {})
    max_position_pct = float(risk_cfg.get("max_position_pct", 0.10))
    stop_loss_pct = float(risk_cfg.get("stop_loss_pct", 0.03))
    daily_max_loss_pct = float(risk_cfg.get("daily_max_loss_pct", 0.04))
    kill_switch_pct = float(risk_cfg.get("kill_switch_drawdown_pct", 0.20))

    # 1. Kill switch
    if peak_equity > 0:
        drawdown = (peak_equity - equity) / peak_equity
        if drawdown >= kill_switch_pct:
            msg = (
                f"KILL SWITCH: drawdown {drawdown:.2%} >= threshold {kill_switch_pct:.2%}"
            )
            logger.critical("[risk] %s | symbol=%s", msg, symbol)
            return False, 0.0, msg

    # 2. Daily loss limit
    daily_start_equity = equity - daily_pnl
    if daily_start_equity > 0:
        daily_pnl_pct = daily_pnl / daily_start_equity
        if daily_pnl_pct <= -daily_max_loss_pct:
            msg = (
                f"Daily loss limit hit: {daily_pnl_pct:.2%} <= -{daily_max_loss_pct:.2%}"
            )
            logger.warning("[risk] %s | symbol=%s", msg, symbol)
            return False, 0.0, msg

    # 3. Position size cap (no price available in this interface — use desired_qty as cap)
    # For backward compat: just return approved at desired_qty if within max_position_pct * equity
    # We don't have current_price here, so we just return desired_qty
    approved_qty = float(int(desired_qty))
    if approved_qty < 1:
        return False, 0.0, "approved_qty < 1"

    msg = f"Approved: qty={approved_qty}"
    return True, approved_qty, msg
