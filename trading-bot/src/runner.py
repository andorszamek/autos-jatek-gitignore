"""Live/paper trading loop: data -> features -> signal -> risk -> order -> state.

run_cycle() executes one full trading cycle:
  1. Load model from models/latest.lgb
  2. Fetch latest bars (lookback=30)
  3. Build features (inference mode)
  4. For each symbol: strategy.decide() → risk check → submit order
  5. Persist state to SQLite (state.db)
  6. Log every decision

SQLite state survives restarts; on startup we reconcile with broker positions.
"""
import json
import logging
import logging.handlers
import sqlite3
from datetime import datetime, timezone
from math import floor
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
_STATE_DB = _ROOT / "state.db"
_MODELS_DIR = _ROOT / "models"
_LOGS_DIR = _ROOT / "logs"

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

class _JsonFormatter(logging.Formatter):
    """Emit log records as JSON lines."""

    def format(self, record: logging.LogRecord) -> str:
        extra = {
            k: v
            for k, v in record.__dict__.items()
            if k not in {
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
                "taskName",
            }
        }
        payload = {
            "time": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "msg": record.getMessage(),
        }
        if extra:
            payload["extra"] = extra
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def setup_logging(log_dir: Path) -> None:
    """Configure JSON structured logging to file + stream to stdout."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
    log_file = log_dir / f"trading_{date_str}.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Remove existing handlers to avoid duplicate output
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)

    json_fmt = _JsonFormatter()

    # File handler (rotating daily)
    fh = logging.FileHandler(str(log_file), encoding="utf-8")
    fh.setFormatter(json_fmt)
    fh.setLevel(logging.DEBUG)
    root_logger.addHandler(fh)

    # Stream handler to stdout
    sh = logging.StreamHandler()
    sh.setFormatter(json_fmt)
    sh.setLevel(logging.INFO)
    root_logger.addHandler(sh)

    logger.info("[runner] Logging initialised. Log file: %s", log_file)


# ---------------------------------------------------------------------------
# SQLite state management
# ---------------------------------------------------------------------------

def _get_db_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_STATE_DB))
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    """Create tables if they don't exist."""
    conn = _get_db_conn()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS positions (
                symbol      TEXT PRIMARY KEY,
                qty         INTEGER NOT NULL,
                avg_price   REAL NOT NULL,
                timestamp   TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS trades (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT NOT NULL,
                symbol      TEXT NOT NULL,
                side        TEXT NOT NULL,
                qty         INTEGER NOT NULL,
                price       REAL NOT NULL,
                cost        REAL NOT NULL DEFAULT 0.0,
                pnl         REAL NOT NULL DEFAULT 0.0
            );

            CREATE TABLE IF NOT EXISTS equity_log (
                timestamp   TEXT PRIMARY KEY,
                equity      REAL NOT NULL,
                cash        REAL NOT NULL
            );
        """)
        conn.commit()
    finally:
        conn.close()


def _load_positions_from_db() -> dict[str, dict]:
    """Load positions table into {symbol: {qty, avg_price}} dict."""
    conn = _get_db_conn()
    try:
        rows = conn.execute("SELECT symbol, qty, avg_price FROM positions").fetchall()
        return {r["symbol"]: {"qty": r["qty"], "avg_price": r["avg_price"]} for r in rows}
    finally:
        conn.close()


def _upsert_position(symbol: str, qty: int, avg_price: float) -> None:
    conn = _get_db_conn()
    try:
        ts = datetime.now(tz=timezone.utc).isoformat()
        if qty == 0:
            conn.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))
        else:
            conn.execute(
                """INSERT INTO positions (symbol, qty, avg_price, timestamp)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(symbol) DO UPDATE SET
                       qty=excluded.qty,
                       avg_price=excluded.avg_price,
                       timestamp=excluded.timestamp""",
                (symbol, qty, avg_price, ts),
            )
        conn.commit()
    finally:
        conn.close()


def _insert_trade(
    symbol: str,
    side: str,
    qty: int,
    price: float,
    cost: float = 0.0,
    pnl: float = 0.0,
) -> None:
    conn = _get_db_conn()
    try:
        ts = datetime.now(tz=timezone.utc).isoformat()
        conn.execute(
            """INSERT INTO trades (timestamp, symbol, side, qty, price, cost, pnl)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (ts, symbol, side, qty, price, cost, pnl),
        )
        conn.commit()
    finally:
        conn.close()


def _log_equity(equity: float, cash: float) -> None:
    conn = _get_db_conn()
    try:
        ts = datetime.now(tz=timezone.utc).isoformat()
        conn.execute(
            """INSERT OR REPLACE INTO equity_log (timestamp, equity, cash)
               VALUES (?, ?, ?)""",
            (ts, equity, cash),
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Main cycle
# ---------------------------------------------------------------------------

def run_cycle(cfg: dict[str, Any], risk_flag: bool = False) -> dict[str, Any]:
    """Execute one full trading cycle.

    Returns summary dict with decisions made this cycle.
    """
    setup_logging(_LOGS_DIR)
    _init_db()

    summary: dict[str, Any] = {
        "cycle_start": datetime.now(tz=timezone.utc).isoformat(),
        "decisions": [],
        "orders_submitted": 0,
        "errors": [],
    }

    symbols: list[str] = cfg.get("universe", [])

    # ------------------------------------------------------------------
    # 1. Load per-symbol models (prefer LSTM, fall back to LightGBM;
    #    per-symbol file first, generic latest.* as fallback)
    # ------------------------------------------------------------------
    models_by_sym: dict = {}
    model_type = "lgbm"

    _has_lstm = (_MODELS_DIR / "latest.lstm").exists() or any(
        (_MODELS_DIR / f"latest_{s}.lstm").exists() for s in symbols
    )
    if _has_lstm:
        model_type = "lstm"
        from src.model_lstm import load as _load_model
    elif (_MODELS_DIR / "latest.lgb").exists() or any(
        (_MODELS_DIR / f"latest_{s}.lgb").exists() for s in symbols
    ):
        from src.model import load as _load_model
    else:
        msg = f"No model found in {_MODELS_DIR}. Run scripts/train.py first."
        logger.error("[runner] %s", msg)
        summary["errors"].append(msg)
        return summary

    ext = "lstm" if model_type == "lstm" else "lgb"
    for sym in symbols:
        sym_path     = _MODELS_DIR / f"latest_{sym}.{ext}"
        generic_path = _MODELS_DIR / f"latest.{ext}"
        path = sym_path if sym_path.exists() else (generic_path if generic_path.exists() else None)
        if path:
            try:
                models_by_sym[sym] = _load_model(str(path))
                logger.info("[runner] Model for %s loaded: %s", sym, path.name)
            except Exception as exc:
                logger.warning("[runner] Failed to load model for %s: %s", sym, exc)

    if not models_by_sym:
        msg = f"No usable model in {_MODELS_DIR}. Run scripts/train.py first."
        logger.error("[runner] %s", msg)
        summary["errors"].append(msg)
        return summary

    logger.info("[runner] Models loaded: %d/%d symbols (type=%s)", len(models_by_sym), len(symbols), model_type)

    # ------------------------------------------------------------------
    # 2. Create broker
    # ------------------------------------------------------------------
    try:
        from src.broker import create_broker
        broker = create_broker(cfg, risk_flag=risk_flag)
        account = broker.get_account()
        equity = account.equity
        cash = account.cash
        logger.info("[runner] Account: equity=%.2f cash=%.2f paper=%s", equity, cash, broker.is_paper())
    except Exception as exc:
        msg = f"Broker error: {exc}"
        logger.error("[runner] %s", msg)
        summary["errors"].append(msg)
        return summary

    # ------------------------------------------------------------------
    # 3. Sync positions: broker vs SQLite
    # ------------------------------------------------------------------
    db_positions = _load_positions_from_db()

    broker_positions: dict[str, dict] = {}
    for sym in symbols:
        try:
            pos = broker.get_position(sym)
            if pos is not None and pos.qty > 0:
                broker_positions[sym] = {
                    "qty": int(pos.qty),
                    "avg_price": float(pos.avg_entry_price),
                }
        except Exception as exc:
            logger.warning("[runner] Could not fetch broker position for %s: %s", sym, exc)

    # Sync broker → SQLite for any discrepancies
    for sym, bpos in broker_positions.items():
        if sym not in db_positions or db_positions[sym]["qty"] != bpos["qty"]:
            logger.info("[runner] Syncing position from broker: %s qty=%d", sym, bpos["qty"])
            _upsert_position(sym, bpos["qty"], bpos["avg_price"])
            db_positions[sym] = bpos

    # ------------------------------------------------------------------
    # 4. Get latest bars and build features (290-day lookback for LSTM)
    # ------------------------------------------------------------------
    lookback = 290
    try:
        from src.data_loader import get_latest_bars
        bars_df = get_latest_bars(symbols, lookback=lookback, cfg=cfg)
        logger.info("[runner] Got %d latest bars for %s", len(bars_df), symbols)
    except Exception as exc:
        msg = f"Data fetch error: {exc}"
        logger.error("[runner] %s", msg)
        summary["errors"].append(msg)
        return summary

    # ------------------------------------------------------------------
    # 5. Risk manager setup
    # ------------------------------------------------------------------
    from src.risk import RiskManager
    risk_mgr = RiskManager(cfg, initial_equity=equity)
    daily_start_equity = equity

    # ------------------------------------------------------------------
    # 6. Build features for all symbols → momentum rank → rotate
    # ------------------------------------------------------------------
    from src.features import build_features
    from src.strategy import BUY_THRESHOLD, SELL_THRESHOLD, momentum_rank, momentum_score

    features_by_sym: dict[str, object] = {}
    price_by_sym: dict[str, float] = {}

    for sym in symbols:
        sym_bars = bars_df[bars_df["symbol"] == sym].copy()
        if sym_bars.empty:
            logger.warning("[runner] No bars for %s", sym)
            continue
        current_price = float(sym_bars["close"].iloc[-1])
        price_by_sym[sym] = current_price
        try:
            X, _ = build_features(sym_bars)
        except Exception as exc:
            logger.warning("[runner] Feature build failed for %s: %s", sym, exc)
            continue
        if not X.empty:
            features_by_sym[sym] = X

    if not features_by_sym:
        summary["errors"].append("No features built for any symbol")
        return summary

    # Dual-momentum: weekly rotation (Mondays) with 2% improvement threshold.
    # On non-rotation days, keep the currently held symbol as target.
    from datetime import date as _date
    is_rotation_day = _date.today().weekday() == 0  # Monday

    held_syms = [s for s, p in db_positions.items() if p.get("qty", 0) > 0]
    currently_held = held_syms[0] if held_syms else None

    if is_rotation_day or currently_held is None:
        new_target = momentum_rank(features_by_sym)
        if currently_held is None or currently_held not in features_by_sym:
            target_sym = new_target
        elif new_target is not None and new_target != currently_held:
            curr_s = momentum_score(features_by_sym.get(currently_held))
            new_s  = momentum_score(features_by_sym.get(new_target))
            target_sym = new_target if new_s > curr_s * 1.02 else currently_held
        else:
            target_sym = currently_held
    else:
        target_sym = currently_held  # Hold through the week, ML can still exit

    # Absolute momentum gate: if top asset 3M return ≤ 0 → go to cash
    if target_sym and target_sym in features_by_sym:
        X_top = features_by_sym[target_sym]
        if "RET_60" in X_top.columns and float(X_top["RET_60"].iloc[-1]) <= 0.0:
            logger.info("[runner] Absolute momentum negative for %s — going to cash", target_sym)
            target_sym = None

    logger.info("[runner] Rotation day=%s target=%s held=%s", is_rotation_day, target_sym, currently_held)

    # ── Rotation: sell any held position that is NOT the target ──────────
    for held_sym in list(db_positions.keys()):
        if held_sym == target_sym:
            continue
        current_qty = db_positions[held_sym]["qty"]
        avg_entry = db_positions[held_sym]["avg_price"]
        current_price = price_by_sym.get(held_sym, avg_entry)
        if current_qty <= 0:
            continue
        logger.info("[runner] ROTATION: selling %s → target is %s", held_sym, target_sym)
        approved, approved_qty, reason = risk_mgr.check_order(
            symbol=held_sym,
            desired_qty=current_qty,
            side="sell",
            current_price=current_price,
            equity=equity,
            daily_start_equity=daily_start_equity,
        )
        if not approved:
            logger.warning("[runner] Rotation sell rejected for %s: %s", held_sym, reason)
            continue
        try:
            order = broker.submit_order(held_sym, approved_qty, "sell")
            pnl = (current_price - avg_entry) * approved_qty
            _insert_trade(held_sym, "sell", approved_qty, current_price, pnl=pnl)
            _upsert_position(held_sym, 0, 0.0)
            db_positions.pop(held_sym, None)
            equity += pnl
            summary["orders_submitted"] += 1
            summary["decisions"].append({
                "symbol": held_sym,
                "signal": "rotation_sell",
                "risk_decision": "approved",
                "order_id": order.id,
            })
        except Exception as exc:
            logger.error("[runner] Rotation sell failed for %s: %s", held_sym, exc)
            summary["errors"].append(f"rotation sell {held_sym}: {exc}")

    # ── Stop-loss check on target position ───────────────────────────────
    if target_sym and target_sym in db_positions:
        current_qty = db_positions[target_sym]["qty"]
        avg_entry = db_positions[target_sym]["avg_price"]
        current_price = price_by_sym.get(target_sym, avg_entry)
        if current_qty > 0 and avg_entry > 0 and risk_mgr.check_stop_loss(target_sym, current_price, avg_entry):
            logger.warning("[runner] Stop-loss triggered for %s — selling", target_sym)
            try:
                order = broker.submit_order(target_sym, current_qty, "sell")
                pnl = (current_price - avg_entry) * current_qty
                _insert_trade(target_sym, "sell", current_qty, current_price, pnl=pnl)
                _upsert_position(target_sym, 0, 0.0)
                db_positions.pop(target_sym, None)
                summary["orders_submitted"] += 1
                summary["decisions"].append({
                    "symbol": target_sym,
                    "signal": "stop_loss",
                    "risk_decision": "approved",
                    "order_id": order.id,
                })
                target_sym = None  # Don't re-enter this cycle
            except Exception as exc:
                logger.error("[runner] Stop-loss order failed for %s: %s", target_sym, exc)
                summary["errors"].append(f"stop_loss order {target_sym}: {exc}")

    # ── ML-gated entry / exit for the target symbol ───────────────────────
    if target_sym and target_sym in features_by_sym:
        X = features_by_sym[target_sym]
        current_price = price_by_sym[target_sym]
        current_pos = db_positions.get(target_sym, {})
        current_qty = current_pos.get("qty", 0)
        avg_entry = current_pos.get("avg_price", 0.0)

        # ML probability — use per-symbol model
        model_for_sym = models_by_sym.get(target_sym)
        proba = 0.5
        if model_for_sym is not None:
            try:
                if model_type == "lstm":
                    from src.model_lstm import predict_proba
                    proba_arr = predict_proba(model_for_sym, X)
                    proba = float(proba_arr[-1]) if len(proba_arr) > 0 else 0.5
                else:
                    from src.model import predict_proba
                    proba = float(predict_proba(model_for_sym, X.tail(1))[0])
            except Exception as exc:
                logger.error("[runner] predict failed for %s: %s — using 0.5", target_sym, exc)

        # Regime
        sma50_vs_sma200 = float(X["SMA50_VS_SMA200"].iloc[-1]) if "SMA50_VS_SMA200" in X.columns else 0.0
        in_golden_cross = sma50_vs_sma200 > 0.0
        in_death_cross  = sma50_vs_sma200 < -0.005
        logger.info(
            "[runner] %s p=%.4f golden=%s death=%s qty=%d",
            target_sym, proba, in_golden_cross, in_death_cross, current_qty,
        )

        action: str | None = None
        desired_qty = 0

        if current_qty > 0 and (in_death_cross or proba < SELL_THRESHOLD):
            action = "sell"
            desired_qty = current_qty
        elif current_qty == 0 and in_golden_cross and proba > BUY_THRESHOLD:
            action = "buy"
            desired_qty = max(1, floor(0.99 * equity / current_price))

        if action:
            approved, approved_qty, reason = risk_mgr.check_order(
                symbol=target_sym,
                desired_qty=desired_qty,
                side=action,
                current_price=current_price,
                equity=equity,
                daily_start_equity=daily_start_equity,
            )
            decision_rec = {
                "symbol": target_sym,
                "signal": action,
                "desired_qty": desired_qty,
                "risk_decision": "approved" if approved else "rejected",
                "reason": reason,
                "order_id": None,
            }
            if not approved:
                logger.info("[runner] Order rejected: %s %s — %s", action, target_sym, reason)
                summary["decisions"].append(decision_rec)
            else:
                try:
                    order = broker.submit_order(target_sym, approved_qty, action)
                    decision_rec["order_id"] = order.id
                    logger.info(
                        "[runner] Order submitted: %s %d %s @ %.4f id=%s",
                        action, approved_qty, target_sym, current_price, order.id,
                    )
                    if action == "buy":
                        new_qty = current_qty + approved_qty
                        new_avg = (avg_entry * current_qty + current_price * approved_qty) / new_qty
                        _upsert_position(target_sym, new_qty, new_avg)
                        _insert_trade(target_sym, "buy", approved_qty, current_price)
                        db_positions[target_sym] = {"qty": new_qty, "avg_price": new_avg}
                    elif action == "sell":
                        pnl = (current_price - avg_entry) * approved_qty
                        _insert_trade(target_sym, "sell", approved_qty, current_price, pnl=pnl)
                        remaining = current_qty - approved_qty
                        if remaining <= 0:
                            _upsert_position(target_sym, 0, 0.0)
                            db_positions.pop(target_sym, None)
                        else:
                            _upsert_position(target_sym, remaining, avg_entry)
                            db_positions[target_sym] = {"qty": remaining, "avg_price": avg_entry}
                    summary["orders_submitted"] += 1
                except Exception as exc:
                    msg = f"Order submission failed: {action} {target_sym}: {exc}"
                    logger.error("[runner] %s", msg)
                    decision_rec["risk_decision"] = "order_failed"
                    decision_rec["error"] = str(exc)
                    summary["errors"].append(msg)
            summary["decisions"].append(decision_rec)
        else:
            logger.info("[runner] HOLD %s (p=%.4f)", target_sym, proba)
            summary["decisions"].append({
                "symbol": target_sym,
                "signal": "hold",
                "risk_decision": "n/a",
                "order_id": None,
            })

    # ------------------------------------------------------------------
    # 7. Update equity log and peak tracker
    # ------------------------------------------------------------------
    try:
        account = broker.get_account()
        equity = account.equity
        cash = account.cash
    except Exception:
        pass

    _log_equity(equity, cash)
    risk_mgr.update_peak(equity)

    summary["cycle_end"] = datetime.now(tz=timezone.utc).isoformat()
    summary["equity"] = equity
    summary["cash"] = cash

    logger.info(
        "[runner] Cycle complete: %d orders, %d decisions, %d errors",
        summary["orders_submitted"],
        len(summary["decisions"]),
        len(summary["errors"]),
    )
    return summary
