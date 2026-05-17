#!/usr/bin/env python3
"""Smoke test: connect to Alpaca paper account and print balance.

Usage:
    python scripts/smoke_test_account.py

Requires a valid .env with ALPACA_API_KEY and ALPACA_SECRET_KEY.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    from src.config import load_config
    from src.broker import create_broker

    cfg = load_config()
    broker = create_broker(cfg, risk_flag=False)

    print(f"Using paper endpoint: {broker.is_paper()}")
    acct = broker.get_account()
    print(f"Account equity:        ${acct.equity:,.2f} {acct.currency}")
    print(f"Cash:                  ${acct.cash:,.2f}")
    print(f"Buying power:          ${acct.buying_power:,.2f}")


if __name__ == "__main__":
    main()
