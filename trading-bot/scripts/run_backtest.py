#!/usr/bin/env python3
"""Run walk-forward backtest and generate report.

Usage:
    python scripts/run_backtest.py [--model lstm|lgbm] [--no-costs] [--refresh]

LSTM (default): trains an LSTM per fold — takes ~5-15 min on M3 Mac.
LightGBM:       legacy tabular model, runs in seconds.

Outputs equity curve CSV and metrics to logs/backtest_<timestamp>/.
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
    parser.add_argument("--model", choices=["lstm", "lgbm"], default="lstm",
                        help="Model architecture (default: lstm)")
    parser.add_argument(
        "--no-costs",
        action="store_true",
        help="Disable cost modelling (sanity check only)",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force re-download data (ignore parquet cache)",
    )
    args = parser.parse_args()

    from src.config import load_config
    cfg = load_config()

    cfg["_model_type"] = args.model

    if args.no_costs:
        print("WARNING: running WITHOUT costs — sanity comparison only.")
        cfg["backtest"]["commission"] = 0.0
        cfg["backtest"]["slippage_pct"] = 0.0
        cfg["backtest"]["fx_cost_pct"] = 0.0

    from src.data_loader import load_history
    from src.backtest import run_backtest

    symbols = cfg["universe"]
    # LSTM needs more history for sufficient training sequences
    if args.model == "lstm":
        days = max(int(cfg.get("history_days", 1500)), 7500)
    else:
        days = int(cfg.get("history_days", 1500))

    print(f"Loading data for backtest: {symbols}, {days} days...")
    df = load_history(
        symbols=symbols,
        timeframe=cfg["timeframe"],
        days=days,
        cfg=cfg,
        force_refresh=args.refresh,
    )

    print(f"Running backtest [{args.model.upper()}]...")
    results = run_backtest(df, cfg)
    print("Backtest complete. Report saved to logs/.")


if __name__ == "__main__":
    main()
