# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Paper-first algorithmic trading bot for SPY (US equities). Trains a LightGBM classifier on Mac, deploys inference to Raspberry Pi. **Real money trading is intentionally hard to enable** — see safety constraints below.

## Commands

```bash
# Setup (inside trading-bot/)
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
brew install libomp          # macOS only — required by LightGBM

# All tests
pytest tests/ -v

# Single test file / single test
pytest tests/test_features.py -v
pytest tests/test_features.py::test_no_lookahead -v

# Full workflow
python scripts/download_data.py          # fetch + cache SPY Parquet
python scripts/train.py                  # train model → models/latest.lgb
python scripts/run_backtest.py           # walk-forward backtest with costs
python scripts/run_backtest.py --refresh # force re-download data
python scripts/run_paper.py              # single paper cycle
python scripts/run_paper.py --loop       # continuous loop
python scripts/smoke_test_account.py     # verify Alpaca connection
```

## Architecture

### Data flow (single cycle)
```
data_loader.get_latest_bars()
  → features.build_features()       # same function used in training AND inference
    → strategy.decide()             # model probability + trend filter → action
      → risk.RiskManager.check_order()  # veto gate — no order bypasses this
        → broker.AlpacaPaperBroker.submit_order()
          → runner._upsert_position()  # SQLite state persisted
```

### Module responsibilities

| Module | Role |
|---|---|
| `config.py` | Loads `config.yaml` + `.env`, validates required keys, exposes `is_live_trading_enabled()` |
| `data_loader.py` | Alpaca primary → yfinance fallback; Parquet cache per symbol; `CANONICAL_COLUMNS` |
| `features.py` | **Single source of truth** for `FEATURE_COLUMNS` — used identically in training and live inference |
| `model.py` | LightGBM train/save/load/predict; time-ordered 80/20 split; saves timestamped + `latest.lgb` |
| `backtest.py` | Walk-forward (5 folds); fresh model per fold; full cost model (commission + slippage + FX) |
| `strategy.py` | `decide()` → `BUY_THRESHOLD=0.52`, `SELL_THRESHOLD=0.48`; SMA200 trend filter |
| `risk.py` | `RiskManager` class: position sizing, stop-loss, daily loss limit, kill switch |
| `runner.py` | `run_cycle()` orchestrates one full cycle; SQLite state in `state.db`; JSON logging |
| `broker.py` | Abstract `Broker` interface; `AlpacaPaperBroker` with double-check at every method call |
| `scheduler.py` | DST-aware NYSE hours; `run_with_health_check()` updates `logs/health.json` |

### Critical invariants — never break these

1. **Paper-first**: `AlpacaPaperBroker` uses paper endpoint unless `LIVE_TRADING=true` env var AND `risk_flag=True` constructor arg are BOTH set. This is checked at every method call, not just `__init__`.

2. **No look-ahead in features**: `next_day_up` target uses `np.where(next_close.notna(), ..., np.nan)` — not a simple boolean comparison (which would silently give `False` for the last row instead of `NaN`, causing the last row to stay in training data with a wrong label).

3. **`FEATURE_COLUMNS` is the single contract** between training and inference. Never add features to `_build_symbol_features()` without adding them to `FEATURE_COLUMNS`, and vice versa.

4. **Backtest must model costs**: `commission + slippage_pct * price * qty + fx_cost_pct * price * qty`. The `--no-costs` flag exists only for sanity comparison.

5. **Risk module has veto power**: `runner.py` never calls `broker.submit_order()` without first calling `risk_mgr.check_order()` and checking `approved is True`.

### Features (14 total, requires ≥200 rows)

Trend: `PRICE_VS_SMA50`, `PRICE_VS_SMA200`, `SMA50_VS_SMA200`
Momentum: `RET_1`, `RET_5`, `RET_20`, `RET_60`
Mean reversion: `RSI_14`, `BB_PCT`
MACD: `MACD_HIST` (normalised by price)
Volatility: `ATR_PCT`, `VOL_20`
Volume: `VOL_RATIO`

SMA200 is the binding constraint — tests need ≥250 synthetic rows to produce valid output.

### State persistence

`state.db` (SQLite, at project root) has three tables:
- `positions`: current holdings (symbol, qty, avg_price)
- `trades`: full trade log
- `equity_log`: equity snapshot per cycle

On restart, `runner.py` syncs broker positions → SQLite to handle discrepancies.

### Two-node deployment

- **Mac**: full `requirements.txt`, runs `train.py` and `run_backtest.py`
- **Pi**: only runtime deps (`alpaca-py pandas pyarrow lightgbm python-dotenv pyyaml yfinance`). Copy `models/latest.lgb` from Mac via `scp`. Triggered by `systemd/trading-bot.timer` at 22:00 UTC daily.

### Live trading gate

```python
# config.py
is_live_trading_enabled(cfg, cli_risk_flag=True)
# Returns True only when BOTH:
#   cfg["_env"]["live_trading"] == True  (LIVE_TRADING=true in .env)
#   cli_risk_flag == True                (--i-understand-the-risk CLI flag)
```

## Config

Key values in `config.yaml` that affect behaviour:

```yaml
universe: ["SPY"]          # symbols to trade
timeframe: "1Day"          # bar size
initial_capital: 1000      # starting capital for backtest
risk:
  max_position_pct: 0.10   # max 10% of equity per position
  stop_loss_pct: 0.03      # 3% stop-loss
  daily_max_loss_pct: 0.04 # halt trading if daily loss >= 4%
  kill_switch_drawdown_pct: 0.20  # close all if drawdown >= 20%
```

## Known issues / history

- `vectorbt` removed from `requirements.txt` — backtest uses custom walk-forward logic, not vectorbt
- macOS requires `brew install libomp` before `lightgbm` can load
- First backtest result (9 basic features, threshold 0.55): CAGR -0.07%, 49 trades/4yr — strategy was too conservative
- Current setup (14 features, threshold 0.52, SMA200 trend filter): improved trade frequency; beat buy-and-hold not guaranteed (SPY bull market is hard to beat)
