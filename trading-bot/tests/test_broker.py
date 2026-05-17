"""Broker safety tests — Phase 1."""
import os
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_broker_paper_by_default(monkeypatch):
    """AlpacaPaperBroker must use paper endpoint when LIVE_TRADING is not set."""
    monkeypatch.setenv("LIVE_TRADING", "false")

    # We can't actually connect to Alpaca, so just verify construction logic.
    # Mock alpaca-py imports to avoid real network calls.
    import unittest.mock as mock
    with mock.patch.dict("sys.modules", {
        "alpaca": mock.MagicMock(),
        "alpaca.trading": mock.MagicMock(),
        "alpaca.trading.client": mock.MagicMock(),
        "alpaca.trading.requests": mock.MagicMock(),
        "alpaca.trading.enums": mock.MagicMock(),
        "alpaca.data": mock.MagicMock(),
        "alpaca.data.historical": mock.MagicMock(),
    }):
        from src.broker import AlpacaPaperBroker
        broker = AlpacaPaperBroker(api_key="k", secret_key="s", risk_flag=False)
        assert broker.is_paper() is True


def test_broker_paper_when_live_env_but_no_flag(monkeypatch):
    """Even with LIVE_TRADING=true, missing risk_flag must force paper."""
    monkeypatch.setenv("LIVE_TRADING", "true")

    import unittest.mock as mock
    with mock.patch.dict("sys.modules", {
        "alpaca": mock.MagicMock(),
        "alpaca.trading": mock.MagicMock(),
        "alpaca.trading.client": mock.MagicMock(),
        "alpaca.trading.requests": mock.MagicMock(),
        "alpaca.trading.enums": mock.MagicMock(),
        "alpaca.data": mock.MagicMock(),
        "alpaca.data.historical": mock.MagicMock(),
    }):
        from src.broker import AlpacaPaperBroker
        broker = AlpacaPaperBroker(api_key="k", secret_key="s", risk_flag=False)
        assert broker.is_paper() is True


def test_broker_live_requires_both(monkeypatch):
    """Live mode requires LIVE_TRADING=true AND risk_flag=True."""
    monkeypatch.setenv("LIVE_TRADING", "true")

    import unittest.mock as mock
    with mock.patch.dict("sys.modules", {
        "alpaca": mock.MagicMock(),
        "alpaca.trading": mock.MagicMock(),
        "alpaca.trading.client": mock.MagicMock(),
        "alpaca.trading.requests": mock.MagicMock(),
        "alpaca.trading.enums": mock.MagicMock(),
        "alpaca.data": mock.MagicMock(),
        "alpaca.data.historical": mock.MagicMock(),
    }):
        from src.broker import AlpacaPaperBroker
        broker = AlpacaPaperBroker(api_key="k", secret_key="s", risk_flag=True)
        assert broker.is_paper() is False
