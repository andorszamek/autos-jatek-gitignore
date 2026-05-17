"""Risk management — position sizing, stop-loss, daily limit, kill switch."""
# Stub — implemented in Phase 6
from typing import Any


def check_order(
    symbol: str,
    desired_qty: float,
    side: str,
    equity: float,
    daily_pnl: float,
    peak_equity: float,
    cfg: dict[str, Any],
) -> tuple[bool, float, str]:
    """Return (approved, approved_qty, reason)."""
    raise NotImplementedError("Implemented in Phase 6")
