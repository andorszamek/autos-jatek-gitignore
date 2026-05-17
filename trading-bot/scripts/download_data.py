#!/usr/bin/env python3
"""Download and cache historical market data.

Usage:
    python scripts/download_data.py [--symbols SPY QQQ] [--days 1500]

Reads universe and history_days from config.yaml by default.
Saves Parquet files to data/<SYMBOL>_<timeframe>.parquet.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download historical market data for the configured universe.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--symbols", nargs="+", help="Override universe symbols")
    parser.add_argument("--days", type=int, help="Override history_days")
    args = parser.parse_args()

    from src.config import load_config
    cfg = load_config()

    symbols = args.symbols or cfg["universe"]
    days = args.days or cfg["history_days"]

    print(f"Downloading {days} days of {cfg['timeframe']} data for: {symbols}")

    from src.data_loader import load_history
    df = load_history(symbols=symbols, timeframe=cfg["timeframe"], days=days, cfg=cfg)
    print(f"Done. {len(df)} rows loaded.")


if __name__ == "__main__":
    main()
