#!/usr/bin/env python3
"""Train the ML model on historical data.

Usage:
    python scripts/train.py [--symbols SPY] [--days 1500]

Loads data from data/, builds features, trains LightGBM, saves model to models/.
Run this on a Mac/desktop — not needed on the Raspberry Pi.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the LightGBM model. Run on Mac, not on Raspberry Pi.",
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

    from src.data_loader import load_history
    from src.features import build_features
    from src.model import train, save

    print(f"Loading data for {symbols} ({days} days)...")
    df = load_history(symbols=symbols, timeframe=cfg["timeframe"], days=days, cfg=cfg)

    print("Building features...")
    X, y = build_features(df)

    print(f"Training on {len(X)} samples...")
    model = train(X, y)

    output_dir = Path(__file__).resolve().parent.parent / "models"
    output_dir.mkdir(exist_ok=True)
    save(model, str(output_dir))
    print("Model saved to models/")


if __name__ == "__main__":
    main()
