"""Risk management tests — Phase 6."""
import logging

import pytest

from src.risk import RiskManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(
    max_position_pct: float = 0.10,
    stop_loss_pct: float = 0.03,
    daily_max_loss_pct: float = 0.04,
    kill_switch_drawdown_pct: float = 0.20,
) -> dict:
    return {
        "risk": {
            "max_position_pct": max_position_pct,
            "stop_loss_pct": stop_loss_pct,
            "daily_max_loss_pct": daily_max_loss_pct,
            "kill_switch_drawdown_pct": kill_switch_drawdown_pct,
        }
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_position_size_capped():
    """Desired 100 shares, but max_position_pct * equity / price = 5 → approved_qty = 5."""
    # equity=10_000, max_position_pct=0.10, price=200 → max_qty = floor(0.10*10000/200) = 5
    cfg = _cfg(max_position_pct=0.10)
    rm = RiskManager(cfg, initial_equity=10_000.0)
    approved, qty, reason = rm.check_order(
        symbol="SPY",
        desired_qty=100,
        side="buy",
        current_price=200.0,
        equity=10_000.0,
        daily_start_equity=10_000.0,
    )
    assert approved is True
    assert qty == 5, f"Expected qty=5, got {qty}"


def test_daily_loss_limit():
    """Equity dropped 5% today, daily_max_loss_pct=0.04 → rejected."""
    cfg = _cfg(daily_max_loss_pct=0.04)
    initial_equity = 10_000.0
    rm = RiskManager(cfg, initial_equity=initial_equity)

    # Daily start equity = 10_000, current equity = 9_500 → -5% loss
    equity = 9_500.0
    daily_start_equity = 10_000.0

    approved, qty, reason = rm.check_order(
        symbol="SPY",
        desired_qty=10,
        side="buy",
        current_price=100.0,
        equity=equity,
        daily_start_equity=daily_start_equity,
    )
    assert approved is False, f"Expected rejection, got approved=True, reason={reason}"
    assert qty == 0
    assert "daily loss" in reason.lower() or "limit" in reason.lower()


def test_kill_switch(caplog):
    """Drawdown 25%, kill_switch_drawdown_pct=0.20 → rejected with CRITICAL log."""
    cfg = _cfg(kill_switch_drawdown_pct=0.20)
    initial_equity = 10_000.0
    rm = RiskManager(cfg, initial_equity=initial_equity)

    # Peak was 10_000, current equity = 7_500 → 25% drawdown
    rm.update_peak(initial_equity)
    equity = 7_500.0

    with caplog.at_level(logging.CRITICAL):
        approved, qty, reason = rm.check_order(
            symbol="SPY",
            desired_qty=10,
            side="buy",
            current_price=100.0,
            equity=equity,
            daily_start_equity=equity,  # no daily loss, only drawdown
        )

    assert approved is False, f"Expected kill-switch rejection, got approved=True"
    assert qty == 0
    assert "kill switch" in reason.lower() or "KILL" in reason

    # Check CRITICAL log was emitted
    critical_msgs = [r for r in caplog.records if r.levelno >= logging.CRITICAL]
    assert len(critical_msgs) > 0, "Expected CRITICAL log for kill switch"


def test_stop_loss_triggered():
    """Entry price 100, current price 96, stop_loss_pct=0.03 → True (4% > 3% threshold)."""
    cfg = _cfg(stop_loss_pct=0.03)
    rm = RiskManager(cfg, initial_equity=10_000.0)

    result = rm.check_stop_loss(
        symbol="SPY",
        current_price=96.0,
        entry_price=100.0,
    )
    assert result is True, "Expected stop-loss to be triggered (4% loss > 3% threshold)"


def test_stop_loss_not_triggered():
    """Entry price 100, current price 98, stop_loss_pct=0.03 → False (2% < 3%)."""
    cfg = _cfg(stop_loss_pct=0.03)
    rm = RiskManager(cfg, initial_equity=10_000.0)

    result = rm.check_stop_loss(
        symbol="SPY",
        current_price=98.0,
        entry_price=100.0,
    )
    assert result is False, "Expected stop-loss NOT triggered (2% loss < 3% threshold)"


def test_normal_order_approved():
    """Normal conditions: within position limit, no loss → approved."""
    cfg = _cfg(
        max_position_pct=0.10,
        daily_max_loss_pct=0.04,
        kill_switch_drawdown_pct=0.20,
    )
    equity = 10_000.0
    rm = RiskManager(cfg, initial_equity=equity)

    # Buy 2 shares at $100 = $200 = 2% of equity → within 10% limit
    approved, qty, reason = rm.check_order(
        symbol="SPY",
        desired_qty=2,
        side="buy",
        current_price=100.0,
        equity=equity,
        daily_start_equity=equity,
    )
    assert approved is True, f"Expected approval, got rejected: {reason}"
    assert qty >= 1


def test_insufficient_equity_for_one_share():
    """If even 1 share costs more than max_position_pct * equity → rejected."""
    cfg = _cfg(max_position_pct=0.01)  # 1% of equity
    rm = RiskManager(cfg, initial_equity=100.0)

    # 1% of $100 = $1, price = $200 → can't even afford 1 share
    approved, qty, reason = rm.check_order(
        symbol="EXPENSIVE",
        desired_qty=1,
        side="buy",
        current_price=200.0,
        equity=100.0,
        daily_start_equity=100.0,
    )
    assert approved is False
    assert qty == 0


def test_update_peak():
    """Peak equity should update when equity rises above previous peak."""
    cfg = _cfg()
    rm = RiskManager(cfg, initial_equity=10_000.0)

    rm.update_peak(12_000.0)
    # Now drawdown from 12_000 with current equity 9_600 = 20% → kill switch
    approved, _, reason = rm.check_order(
        symbol="SPY",
        desired_qty=1,
        side="buy",
        current_price=100.0,
        equity=9_600.0,
        daily_start_equity=9_600.0,
    )
    assert approved is False
    assert "kill switch" in reason.lower() or "KILL" in reason


def test_sell_order_position_size_check():
    """Sell orders should also respect position size cap."""
    cfg = _cfg(max_position_pct=0.10)
    rm = RiskManager(cfg, initial_equity=10_000.0)

    # Want to sell 100 shares, but max_qty = floor(0.10 * 10000 / 100) = 10
    approved, qty, reason = rm.check_order(
        symbol="SPY",
        desired_qty=100,
        side="sell",
        current_price=100.0,
        equity=10_000.0,
        daily_start_equity=10_000.0,
    )
    assert approved is True
    assert qty == 10
