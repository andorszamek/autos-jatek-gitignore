"""Config loading tests — Phase 1."""
import os
import tempfile
from pathlib import Path

import pytest
import yaml


def _write_valid_config(path: Path) -> None:
    cfg = {
        "universe": ["SPY"],
        "timeframe": "1Day",
        "initial_capital": 1000,
        "history_days": 1500,
        "model": {"type": "lightgbm", "target": "next_day_up"},
        "risk": {
            "max_position_pct": 0.10,
            "stop_loss_pct": 0.03,
            "daily_max_loss_pct": 0.04,
            "kill_switch_drawdown_pct": 0.20,
        },
        "backtest": {
            "commission": 0.0,
            "slippage_pct": 0.0002,
            "fx_cost_pct": 0.005,
            "train_test_split": "walk_forward",
        },
    }
    path.write_text(yaml.dump(cfg))


def _write_valid_env(path: Path) -> None:
    path.write_text("ALPACA_API_KEY=test_key\nALPACA_SECRET_KEY=test_secret\n")


def test_load_config_success(tmp_path, monkeypatch):
    cfg_file = tmp_path / "config.yaml"
    env_file = tmp_path / ".env"
    _write_valid_config(cfg_file)
    _write_valid_env(env_file)

    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.config import load_config

    cfg = load_config(env_path=env_file, config_path=cfg_file)
    assert cfg["universe"] == ["SPY"]
    assert cfg["_env"]["alpaca_api_key"] == "test_key"
    assert cfg["_env"]["live_trading"] is False


def test_load_config_missing_env_key(tmp_path, monkeypatch):
    cfg_file = tmp_path / "config.yaml"
    env_file = tmp_path / ".env"
    _write_valid_config(cfg_file)
    env_file.write_text("ALPACA_API_KEY=only_key\n")  # missing SECRET

    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)

    from src.config import load_config
    with pytest.raises(SystemExit):
        load_config(env_path=env_file, config_path=cfg_file)


def test_load_config_missing_yaml_key(tmp_path, monkeypatch):
    cfg_file = tmp_path / "config.yaml"
    env_file = tmp_path / ".env"
    cfg_file.write_text(yaml.dump({"universe": ["SPY"]}))  # missing most keys
    _write_valid_env(env_file)

    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)

    from src.config import load_config
    with pytest.raises(SystemExit):
        load_config(env_path=env_file, config_path=cfg_file)


def test_is_live_trading_disabled_by_default(tmp_path, monkeypatch):
    cfg_file = tmp_path / "config.yaml"
    env_file = tmp_path / ".env"
    _write_valid_config(cfg_file)
    _write_valid_env(env_file)

    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    monkeypatch.delenv("LIVE_TRADING", raising=False)

    from src.config import load_config, is_live_trading_enabled
    cfg = load_config(env_path=env_file, config_path=cfg_file)
    assert is_live_trading_enabled(cfg, cli_risk_flag=True) is False


def test_is_live_trading_requires_both_flags(tmp_path, monkeypatch):
    cfg_file = tmp_path / "config.yaml"
    env_file = tmp_path / ".env"
    _write_valid_config(cfg_file)
    env_file.write_text(
        "ALPACA_API_KEY=k\nALPACA_SECRET_KEY=s\nLIVE_TRADING=true\n"
    )

    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    monkeypatch.delenv("LIVE_TRADING", raising=False)

    from src.config import load_config, is_live_trading_enabled
    cfg = load_config(env_path=env_file, config_path=cfg_file)

    # LIVE_TRADING=true but NO cli flag -> still False
    assert is_live_trading_enabled(cfg, cli_risk_flag=False) is False
    # LIVE_TRADING=true AND cli flag -> True
    assert is_live_trading_enabled(cfg, cli_risk_flag=True) is True
