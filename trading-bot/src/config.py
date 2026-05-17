"""Load and validate config.yaml + .env."""
import os
import sys
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _ROOT / "config.yaml"
_ENV_PATH = _ROOT / ".env"

_REQUIRED_ENV = ["ALPACA_API_KEY", "ALPACA_SECRET_KEY"]
_REQUIRED_CONFIG_KEYS = [
    "universe",
    "timeframe",
    "initial_capital",
    "history_days",
    "model",
    "risk",
    "backtest",
]


def load_config(env_path: Path | None = None, config_path: Path | None = None) -> dict[str, Any]:
    """Load config.yaml and .env, validate required fields, return merged config dict."""
    load_dotenv(dotenv_path=env_path or _ENV_PATH, override=False)

    cfg_path = config_path or _CONFIG_PATH
    if not cfg_path.exists():
        _die(f"config.yaml not found at {cfg_path}")

    with open(cfg_path) as f:
        cfg: dict[str, Any] = yaml.safe_load(f) or {}

    missing_cfg = [k for k in _REQUIRED_CONFIG_KEYS if k not in cfg]
    if missing_cfg:
        _die(f"config.yaml is missing required keys: {missing_cfg}")

    missing_env = [k for k in _REQUIRED_ENV if not os.getenv(k)]
    if missing_env:
        _die(
            f"Missing required environment variables: {missing_env}\n"
            f"Copy .env.example to .env and fill in your Alpaca credentials."
        )

    cfg["_env"] = {
        "alpaca_api_key": os.environ["ALPACA_API_KEY"],
        "alpaca_secret_key": os.environ["ALPACA_SECRET_KEY"],
        "alpaca_paper": os.getenv("ALPACA_PAPER", "true").lower() == "true",
        "live_trading": os.getenv("LIVE_TRADING", "false").lower() == "true",
    }

    return cfg


def is_live_trading_enabled(cfg: dict[str, Any], cli_risk_flag: bool = False) -> bool:
    """Return True only when LIVE_TRADING=true AND --i-understand-the-risk flag is present."""
    return cfg["_env"]["live_trading"] and cli_risk_flag


def _die(msg: str) -> None:
    print(f"[config] ERROR: {msg}", file=sys.stderr)
    sys.exit(1)
