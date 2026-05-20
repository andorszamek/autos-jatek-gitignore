#!/usr/bin/env python3
"""Train the ML model on historical data.

Usage:
    python scripts/train.py [--model lstm|lgbm] [--symbols SPY] [--days N]

LSTM (default): downloads 7500 days (~30 years), trains on Mac with MPS.
LightGBM:       legacy tabular model, 1500 days.

Saves model to models/latest.lstm (or .lgb).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train ML model for SPY directional prediction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model", choices=["lstm", "lgbm"], default="lstm",
                        help="Model architecture (default: lstm)")
    parser.add_argument("--symbols", nargs="+", help="Override universe symbols")
    parser.add_argument("--days", type=int, help="Override history_days")
    args = parser.parse_args()

    from src.config import load_config
    cfg = load_config()

    symbols = args.symbols or cfg["universe"]

    if args.model == "lstm":
        from src.model_lstm import train, save
        days = args.days or max(int(cfg.get("history_days", 1500)), 7500)
    else:
        from src.model import train, save
        days = args.days or int(cfg.get("history_days", 1500))

    from src.data_loader import load_history
    from src.features import build_features

    print(f"Loading data for {symbols} ({days} days)...")
    df = load_history(symbols=symbols, timeframe=cfg["timeframe"], days=days, cfg=cfg)

    print("Building features...")
    X, y = build_features(df)

    print(f"Training [{args.model.upper()}] on {len(X)} samples...")
    model = train(X, y, cfg)

    output_dir = Path(__file__).resolve().parent.parent / "models"
    output_dir.mkdir(exist_ok=True)
    save(model, str(output_dir))
    print("Model saved to models/")


if __name__ == "__main__":
    main()
