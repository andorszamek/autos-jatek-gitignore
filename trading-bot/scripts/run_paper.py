#!/usr/bin/env python3
"""Start the paper trading bot.

Usage:
    python scripts/run_paper.py [--loop] [--interval 60]

    --loop         Run continuously (default: single cycle)
    --interval N   Seconds between cycles when --loop is used (default: 60)

LIVE TRADING WARNING:
    This script NEVER enables live trading on its own.
    Live trading requires BOTH:
      - LIVE_TRADING=true in .env
      - --i-understand-the-risk CLI flag
    Even then, carefully review the risk parameters in config.yaml first.
    You can lose real money. Paper trade for months before going live.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run paper trading bot (default) or live with explicit risk flag.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--loop", action="store_true", help="Run continuously")
    parser.add_argument("--interval", type=int, default=60, help="Seconds between cycles")
    parser.add_argument(
        "--i-understand-the-risk",
        action="store_true",
        dest="risk_flag",
        help="Required (with LIVE_TRADING=true in .env) to enable live trading",
    )
    args = parser.parse_args()

    from src.config import load_config, is_live_trading_enabled
    cfg = load_config()

    live = is_live_trading_enabled(cfg, cli_risk_flag=args.risk_flag)
    if live:
        print("=" * 60)
        print("LIVE TRADING ENABLED — REAL MONEY AT RISK")
        print("=" * 60)
    else:
        print("Paper trading mode. No real money will be used.")

    from src.runner import run_cycle

    if args.loop:
        print(f"Running in loop mode (interval={args.interval}s). Ctrl+C to stop.")
        while True:
            run_cycle(cfg, risk_flag=args.risk_flag)
            time.sleep(args.interval)
    else:
        run_cycle(cfg, risk_flag=args.risk_flag)


if __name__ == "__main__":
    main()
