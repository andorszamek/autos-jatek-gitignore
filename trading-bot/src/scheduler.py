"""Scheduler for Raspberry Pi — market-hours aware cron/systemd helper.

Provides:
  is_market_open()         — NYSE hours 9:30–16:00 ET, Mon–Fri
  should_run_today()       — daily strategy: run after market close (16:30 ET)
  get_next_run_time()      — next scheduled run datetime
  run_with_health_check()  — run one cycle, update logs/health.json
"""
import json
import logging
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
_LOGS_DIR = _ROOT / "logs"
_HEALTH_FILE = _LOGS_DIR / "health.json"

# NYSE timezone offset: EST = UTC-5, EDT = UTC-4
# We use a simple fixed offset approach to avoid requiring pytz on the Pi.
# The systemd timer fires at 22:00 UTC (UTC+5 in winter = 17:00 ET, well after close).
# For the is_market_open() check we convert UTC → ET manually.

_ET_OFFSET_WINTER = timedelta(hours=-5)  # EST (Nov–Mar)
_ET_OFFSET_SUMMER = timedelta(hours=-4)  # EDT (Mar–Nov)

# NYSE trading hours (ET)
_MARKET_OPEN_ET = time(9, 30)
_MARKET_CLOSE_ET = time(16, 0)

# Run-after-close time (ET)
_RUN_AFTER_ET = time(16, 30)


def _is_dst(dt_utc: datetime) -> bool:
    """Approximate US DST: second Sunday in March → first Sunday in November."""
    year = dt_utc.year
    # Second Sunday in March
    march_1 = datetime(year, 3, 1, tzinfo=timezone.utc)
    first_sunday_march = march_1 + timedelta(days=(6 - march_1.weekday()) % 7)
    dst_start = first_sunday_march + timedelta(weeks=1)  # second Sunday

    # First Sunday in November
    nov_1 = datetime(year, 11, 1, tzinfo=timezone.utc)
    dst_end = nov_1 + timedelta(days=(6 - nov_1.weekday()) % 7)

    # DST starts at 2am ET (7am UTC)
    dst_start_utc = datetime(year, dst_start.month, dst_start.day, 7, 0, tzinfo=timezone.utc)
    dst_end_utc = datetime(year, dst_end.month, dst_end.day, 6, 0, tzinfo=timezone.utc)

    return dst_start_utc <= dt_utc < dst_end_utc


def _utc_to_et(dt_utc: datetime) -> datetime:
    """Convert UTC datetime to Eastern Time (handles DST)."""
    offset = _ET_OFFSET_SUMMER if _is_dst(dt_utc) else _ET_OFFSET_WINTER
    return dt_utc + offset


def _is_market_holiday(dt_et: datetime) -> bool:
    """Simple NYSE holiday check for the most common fixed/observed holidays.

    Not exhaustive — use pandas_market_calendars for production-grade logic.
    """
    year = dt_et.year
    month = dt_et.month
    day = dt_et.day
    weekday = dt_et.weekday()  # 0=Mon, 6=Sun

    # New Year's Day (observed)
    if month == 1 and day == 1:
        return True
    # MLK Day: 3rd Monday in January
    if month == 1 and weekday == 0 and 15 <= day <= 21:
        return True
    # Presidents' Day: 3rd Monday in February
    if month == 2 and weekday == 0 and 15 <= day <= 21:
        return True
    # Good Friday (approximate: Friday before Easter — not computed here)
    # Independence Day (July 4)
    if month == 7 and day == 4:
        return True
    if month == 7 and day == 5 and weekday == 0:  # observed Monday
        return True
    if month == 7 and day == 3 and weekday == 4:  # observed Friday
        return True
    # Labor Day: 1st Monday in September
    if month == 9 and weekday == 0 and 1 <= day <= 7:
        return True
    # Thanksgiving: 4th Thursday in November
    if month == 11 and weekday == 3 and 22 <= day <= 28:
        return True
    # Christmas
    if month == 12 and day == 25:
        return True
    if month == 12 and day == 26 and weekday == 0:  # observed Monday
        return True
    if month == 12 and day == 24 and weekday == 4:  # observed Friday
        return True

    return False


def is_market_open() -> bool:
    """Return True if NYSE is currently open (9:30–16:00 ET, Mon–Fri, excluding holidays)."""
    now_utc = datetime.now(tz=timezone.utc)
    now_et = _utc_to_et(now_utc)

    # Weekend
    if now_et.weekday() >= 5:  # Saturday=5, Sunday=6
        return False

    # Holiday
    if _is_market_holiday(now_et):
        return False

    # Trading hours
    t = now_et.time()
    return _MARKET_OPEN_ET <= t < _MARKET_CLOSE_ET


def should_run_today() -> bool:
    """Return True if it's a valid trading day and we're past 16:30 ET.

    For daily-bar strategy: run once per day after market close.
    """
    now_utc = datetime.now(tz=timezone.utc)
    now_et = _utc_to_et(now_utc)

    if now_et.weekday() >= 5:
        return False

    if _is_market_holiday(now_et):
        return False

    t = now_et.time()
    return t >= _RUN_AFTER_ET


def get_next_run_time() -> datetime:
    """Return the next scheduled run datetime (in UTC).

    Runs at 16:30 ET on the next valid trading day.
    """
    now_utc = datetime.now(tz=timezone.utc)

    # Try today first, then look ahead up to 7 days
    for days_ahead in range(8):
        candidate_utc = now_utc + timedelta(days=days_ahead)
        candidate_et = _utc_to_et(candidate_utc)

        if candidate_et.weekday() >= 5:
            continue
        if _is_market_holiday(candidate_et):
            continue

        # Build the run time for this candidate date (16:30 ET)
        run_et = datetime(
            candidate_et.year,
            candidate_et.month,
            candidate_et.day,
            _RUN_AFTER_ET.hour,
            _RUN_AFTER_ET.minute,
            tzinfo=timezone.utc,  # placeholder — will convert back
        )
        # Convert 16:30 ET to UTC
        offset = _ET_OFFSET_SUMMER if _is_dst(candidate_utc) else _ET_OFFSET_WINTER
        run_utc = run_et - offset  # ET + |offset| = UTC

        if run_utc > now_utc:
            return run_utc

    # Fallback: tomorrow
    return now_utc + timedelta(days=1)


def _update_health(last_run: str, success: bool, error: str | None = None) -> None:
    """Write health.json with last_run, last_success, last_error."""
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        if _HEALTH_FILE.exists():
            with open(_HEALTH_FILE) as f:
                health = json.load(f)
        else:
            health = {}
    except Exception:
        health = {}

    health["last_run"] = last_run
    if success:
        health["last_success"] = last_run
        health["last_error"] = None
    else:
        health["last_error"] = error or "unknown error"

    with open(_HEALTH_FILE, "w") as f:
        json.dump(health, f, indent=2)

    logger.debug("[scheduler] Health file updated: %s", _HEALTH_FILE)


def run_with_health_check(cfg: dict[str, Any], risk_flag: bool = False) -> None:
    """Run one trading cycle and update logs/health.json.

    Handles exceptions gracefully — health file always updated.
    """
    from src.runner import run_cycle

    now_utc = datetime.now(tz=timezone.utc).isoformat()
    logger.info("[scheduler] Starting cycle at %s", now_utc)

    try:
        summary = run_cycle(cfg, risk_flag=risk_flag)
        errors = summary.get("errors", [])
        success = len(errors) == 0
        error_msg = "; ".join(errors) if errors else None
        _update_health(now_utc, success=success, error=error_msg)
        logger.info(
            "[scheduler] Cycle complete. orders=%d errors=%d",
            summary.get("orders_submitted", 0),
            len(errors),
        )
    except Exception as exc:
        error_msg = str(exc)
        logger.error("[scheduler] Cycle failed: %s", error_msg, exc_info=True)
        _update_health(now_utc, success=False, error=error_msg)
        raise
