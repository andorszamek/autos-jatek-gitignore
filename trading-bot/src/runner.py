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

    # ------------------------------------------------------------------
    # 1. Load model (lazy import to avoid training deps on Pi)
    # ------------------------------------------------------------------
    model_path = _MODELS_DIR / "latest.lgb"
    if not model_path.exists():
        msg = f"Model not found at {model_path}. Run scripts/train.py first."
        logger.error("[runner] %s", msg)
        summary["errors"].append(msg)
        return summary

    try:
        from src.model import load as model_load
        model = model_load(str(model_path))
        logger.info("[runner] Model loaded from %s", model_path)
    except Exception as exc:
        msg = f"Failed to load model: {exc}"
        logger.error("[runner] %s", msg)
        summary["errors"].append(msg)
        return summary

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
    symbols: list[str] = cfg.get("universe", [])

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
    # 4. Get latest bars and build features
    # ------------------------------------------------------------------
    lookback = 30
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

    # Daily start equity: approximate as current equity (first cycle of the day)
    daily_start_equity = equity

    # ------------------------------------------------------------------
    # 6. Per-symbol: features → signal → risk → order
    # ------------------------------------------------------------------
    from src.features import build_features, FEATURE_COLUMNS
    from src.strategy import decide

    for sym in symbols:
        sym_bars = bars_df[bars_df["symbol"] == sym].copy()
        if sym_bars.empty:
            logger.warning("[runner] No bars for %s", sym)
            continue

        # Build features (inference mode)
        try:
            X, _ = build_features(sym_bars)
        except Exception as exc:
            logger.warning("[runner] Feature build failed for %s: %s", sym, exc)
            continue

        if X.empty:
            logger.warning("[runner] Empty features for %s", sym)
            continue

        # Current position
        current_pos = db_positions.get(sym, {})
        current_qty = current_pos.get("qty", 0)
        avg_entry = current_pos.get("avg_price", 0.0)

        # Current price (last close)
        current_price = float(sym_bars["close"].iloc[-1])

        # Check stop-loss first
        if current_qty > 0 and avg_entry > 0:
            if risk_mgr.check_stop_loss(sym, current_price, avg_entry):
                logger.warning("[runner] Stop-loss triggered for %s — selling", sym)
                try:
                    order = broker.submit_order(sym, current_qty, "sell")
                    pnl = (current_price - avg_entry) * current_qty
                    _insert_trade(sym, "sell", current_qty, current_price, pnl=pnl)
                    _upsert_position(sym, 0, 0.0)
                    db_positions.pop(sym, None)
                    summary["orders_submitted"] += 1
                    summary["decisions"].append({
                        "symbol": sym,
                        "signal": "stop_loss",
                        "risk_decision": "approved",
                        "order_id": order.id,
                    })
                except Exception as exc:
                    logger.error("[runner] Stop-loss order failed for %s: %s", sym, exc)
                    summary["errors"].append(f"stop_loss order {sym}: {exc}")
                continue

        # Strategy signal
        action, desired_qty = decide(X, current_qty, model, cfg)

        if action == "hold":
            logger.info("[runner] HOLD %s", sym)
            summary["decisions"].append({
                "symbol": sym,
                "signal": "hold",
                "risk_decision": "n/a",
                "order_id": None,
            })
            continue

        # Finalize quantity for buy using price + equity
        if action == "buy":
            desired_qty = max(1, floor(
                float(cfg.get("risk", {}).get("max_position_pct", 0.10)) * equity / current_price
            ))

        # Risk check
        approved, approved_qty, reason = risk_mgr.check_order(
            symbol=sym,
            desired_qty=desired_qty,
            side=action,
            current_price=current_price,
            equity=equity,
            daily_start_equity=daily_start_equity,
        )

        decision_rec = {
            "symbol": sym,
            "signal": action,
            "desired_qty": desired_qty,
            "risk_decision": "approved" if approved else "rejected",
            "reason": reason,
            "order_id": None,
        }

        if not approved:
            logger.info("[runner] Order rejected: %s %s — %s", action, sym, reason)
            summary["decisions"].append(decision_rec)
            continue

        # Submit order
        try:
            order = broker.submit_order(sym, approved_qty, action)
            decision_rec["order_id"] = order.id
            logger.info(
                "[runner] Order submitted: %s %d %s @ %.4f id=%s",
                action, approved_qty, sym, current_price, order.id,
            )

            # Update SQLite state
            if action == "buy":
                new_qty = current_qty + approved_qty
                new_avg = (
                    (avg_entry * current_qty + current_price * approved_qty) / new_qty
                    if new_qty > 0 else current_price
                )
                _upsert_position(sym, new_qty, new_avg)
                _insert_trade(sym, "buy", approved_qty, current_price)
                db_positions[sym] = {"qty": new_qty, "avg_price": new_avg}

            elif action == "sell":
                pnl = (current_price - avg_entry) * approved_qty
                _insert_trade(sym, "sell", approved_qty, current_price, pnl=pnl)
                remaining = current_qty - approved_qty
                if remaining <= 0:
                    _upsert_position(sym, 0, 0.0)
                    db_positions.pop(sym, None)
                else:
                    _upsert_position(sym, remaining, avg_entry)
                    db_positions[sym] = {"qty": remaining, "avg_price": avg_entry}

            summary["orders_submitted"] += 1

        except Exception as exc:
            msg = f"Order submission failed: {action} {sym}: {exc}"
            logger.error("[runner] %s", msg)
            decision_rec["risk_decision"] = "order_failed"
            decision_rec["error"] = str(exc)
            summary["errors"].append(msg)

        summary["decisions"].append(decision_rec)

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
