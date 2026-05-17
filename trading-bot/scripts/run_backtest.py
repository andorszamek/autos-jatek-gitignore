#!/usr/bin/env python3
"""Run walk-forward backtest and generate report.

Usage:
    python scripts/run_backtest.py [--no-costs]

Outputs equity curve CSV and metrics to logs/backtest_<timestamp>/.
Always includes commission, slippage, and FX costs unless --no-costs is given
(use --no-costs only for sanity comparison, not for real evaluation).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run walk-forward backtest with cost modelling.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--no-costs",
        action="store_true",
        help="Disable cost modelling (sanity check only — not for real evaluation)",
    )
    args = parser.parse_args()

    from src.config import load_config
    cfg = load_config()

    if args.no_costs:
        print("WARNING: running WITHOUT costs — this is for sanity comparison only.")
        cfg["backtest"]["commission"] = 0.0
        cfg["backtest"]["slippage_pct"] = 0.0
        cfg["backtest"]["fx_cost_pct"] = 0.0

    from src.data_loader import load_history
    from src.backtest import run_backtest

    symbols = cfg["universe"]
    days = cfg["history_days"]

    print(f"Loading data for backtest: {symbols}, {days} days...")
    df = load_history(symbols=symbols, timeframe=cfg["timeframe"], days=days, cfg=cfg)

    print("Running backtest...")
    results = run_backtest(df, cfg)
    print("Backtest complete. Report saved to logs/.")


if __name__ == "__main__":
    main()
