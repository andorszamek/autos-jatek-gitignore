"""Historical and live market data. Canonical DataFrame format shared by all modules."""
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

CANONICAL_COLUMNS = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]

_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _ROOT / "data"


def _data_path(symbol: str, timeframe: str) -> Path:
    _DATA_DIR.mkdir(exist_ok=True)
    return _DATA_DIR / f"{symbol}_{timeframe}.parquet"


def _validate_df(df: pd.DataFrame, context: str) -> pd.DataFrame:
    """Log and drop duplicate timestamps; log NaN counts; raise if empty."""
    if df.empty:
        raise ValueError(f"[data_loader] {context}: DataFrame is empty after loading.")

    nan_counts = df[["open", "high", "low", "close", "volume"]].isna().sum()
    if nan_counts.any():
        logger.warning("[data_loader] %s NaN counts:\n%s", context, nan_counts[nan_counts > 0])

    before = len(df)
    df = df.drop_duplicates(subset=["timestamp", "symbol"])
    if len(df) < before:
        logger.warning(
            "[data_loader] %s: dropped %d duplicate (timestamp, symbol) rows",
            context,
            before - len(df),
        )

    if df.empty:
        raise ValueError(f"[data_loader] {context}: DataFrame is empty after deduplication.")

    return df


def _fetch_alpaca(
    symbols: list[str],
    timeframe: str,
    start: datetime,
    end: datetime,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """Fetch historical bars from Alpaca."""
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    api_key = cfg["_env"]["alpaca_api_key"]
    secret_key = cfg["_env"]["alpaca_secret_key"]
    client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)

    # Parse timeframe string like "1Day", "1Hour", "5Min"
    tf_map = {
        "1day": TimeFrame.Day,
        "1hour": TimeFrame.Hour,
        "1min": TimeFrame.Minute,
        "5min": TimeFrame(5, TimeFrameUnit.Minute),
        "15min": TimeFrame(15, TimeFrameUnit.Minute),
    }
    tf_key = timeframe.lower()
    tf_obj = tf_map.get(tf_key, TimeFrame.Day)

    request = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=tf_obj,
        start=start,
        end=end,
        feed="iex",
    )
    bars = client.get_stock_bars(request)
    df_raw = bars.df

    if df_raw.empty:
        raise ValueError("Alpaca returned no data.")

    df_raw = df_raw.reset_index()

    # Alpaca multi-symbol returns MultiIndex (symbol, timestamp) or columns 'symbol'+'timestamp'
    if "symbol" not in df_raw.columns:
        df_raw["symbol"] = symbols[0] if len(symbols) == 1 else "UNKNOWN"

    # Normalise column names
    col_map = {
        "t": "timestamp",
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
    }
    df_raw.rename(columns=col_map, inplace=True)

    # Ensure timestamp column
    if "timestamp" not in df_raw.columns:
        # Try index
        df_raw = df_raw.reset_index()
        if "timestamp" not in df_raw.columns:
            # Alpaca may use 'time'
            for possible in ["time", "date", "index"]:
                if possible in df_raw.columns:
                    df_raw.rename(columns={possible: "timestamp"}, inplace=True)
                    break

    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], utc=True)

    for col in ["open", "high", "low", "close"]:
        if col in df_raw.columns:
            df_raw[col] = df_raw[col].astype(float)
    if "volume" in df_raw.columns:
        df_raw["volume"] = df_raw["volume"].astype(float)

    return df_raw[CANONICAL_COLUMNS]


def _fetch_yfinance(
    symbols: list[str],
    timeframe: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Fallback: fetch from yfinance."""
    import yfinance as yf

    interval_map = {
        "1day": "1d",
        "1hour": "1h",
        "1min": "1m",
        "5min": "5m",
        "15min": "15m",
    }
    interval = interval_map.get(timeframe.lower(), "1d")

    frames = []
    for sym in symbols:
        ticker = yf.Ticker(sym)
        hist = ticker.history(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval=interval,
            auto_adjust=True,
        )
        if hist.empty:
            logger.warning("[data_loader] yfinance returned no data for %s", sym)
            continue
        hist = hist.reset_index()
        hist["symbol"] = sym

        ts_col = "Date" if "Date" in hist.columns else "Datetime"
        hist.rename(
            columns={
                ts_col: "timestamp",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            },
            inplace=True,
        )
        hist["timestamp"] = pd.to_datetime(hist["timestamp"], utc=True)
        frames.append(hist[CANONICAL_COLUMNS])

    if not frames:
        raise ValueError(f"yfinance returned no data for symbols: {symbols}")

    return pd.concat(frames, ignore_index=True)


def load_history(
    symbols: list[str],
    timeframe: str,
    days: int,
    cfg: dict[str, Any] | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Load historical OHLCV data, with parquet cache.

    Primary source: Alpaca StockHistoricalDataClient.
    Fallback: yfinance.
    Cache: data/<SYMBOL>_<timeframe>.parquet per symbol.

    Returns canonical DataFrame with columns: timestamp, symbol, open, high, low, close, volume.
    """
    end = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=days)

    frames: list[pd.DataFrame] = []

    for sym in symbols:
        cache_path = _data_path(sym, timeframe)

        if not force_refresh and cache_path.exists():
            logger.info("[data_loader] Loading %s from cache: %s", sym, cache_path)
            try:
                cached = pd.read_parquet(cache_path)
                # Ensure timestamp is tz-aware
                if "timestamp" in cached.columns:
                    cached["timestamp"] = pd.to_datetime(cached["timestamp"], utc=True)
                frames.append(cached)
                continue
            except Exception as exc:
                logger.warning("[data_loader] Cache read failed for %s: %s — re-fetching", sym, exc)

        logger.info("[data_loader] Fetching %s from remote (days=%d)", sym, days)
        sym_df: pd.DataFrame | None = None

        # Primary: Alpaca
        if cfg is not None and "_env" in cfg:
            try:
                sym_df = _fetch_alpaca([sym], timeframe, start, end, cfg)
                logger.info("[data_loader] Alpaca fetch OK for %s: %d rows", sym, len(sym_df))
            except Exception as exc:
                logger.warning("[data_loader] Alpaca failed for %s: %s — trying yfinance", sym, exc)

        # Fallback: yfinance
        if sym_df is None or sym_df.empty:
            logger.info("[data_loader] Using yfinance for %s", sym)
            try:
                sym_df = _fetch_yfinance([sym], timeframe, start, end)
                logger.info("[data_loader] yfinance fetch OK for %s: %d rows", sym, len(sym_df))
            except Exception as exc:
                raise RuntimeError(
                    f"Both Alpaca and yfinance failed for {sym}"
                ) from exc

        # Validate and cache
        sym_df = _validate_df(sym_df, sym)
        sym_df = sym_df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

        try:
            sym_df.to_parquet(cache_path, index=False)
            logger.info("[data_loader] Cached %s to %s", sym, cache_path)
        except Exception as exc:
            logger.warning("[data_loader] Failed to cache %s: %s", sym, exc)

        frames.append(sym_df)

    if not frames:
        raise ValueError(f"No data loaded for symbols: {symbols}")

    combined = pd.concat(frames, ignore_index=True)
    combined = _validate_df(combined, "combined")
    combined = combined.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    return combined


def get_latest_bars(
    symbols: list[str],
    lookback: int,
    cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Fetch the most recent `lookback` bars for each symbol — no cache write.

    Returns canonical DataFrame.
    """
    timeframe = (cfg or {}).get("timeframe", "1Day")
    end = datetime.now(tz=timezone.utc)
    # Add buffer for weekends/holidays
    start = end - timedelta(days=lookback * 2 + 10)

    frames: list[pd.DataFrame] = []

    for sym in symbols:
        sym_df: pd.DataFrame | None = None

        # Primary: Alpaca
        if cfg is not None and "_env" in cfg:
            try:
                sym_df = _fetch_alpaca([sym], timeframe, start, end, cfg)
            except Exception as exc:
                logger.warning(
                    "[data_loader] get_latest_bars Alpaca failed for %s: %s — trying yfinance",
                    sym, exc,
                )

        # Fallback: yfinance
        if sym_df is None or sym_df.empty:
            try:
                sym_df = _fetch_yfinance([sym], timeframe, start, end)
            except Exception as exc:
                raise RuntimeError(
                    f"Both Alpaca and yfinance failed for {sym} (get_latest_bars)"
                ) from exc

        sym_df = sym_df.sort_values("timestamp").reset_index(drop=True)
        # Keep only the last `lookback` rows
        sym_df = sym_df.tail(lookback).reset_index(drop=True)
        frames.append(sym_df)

    if not frames:
        raise ValueError(f"No latest bars loaded for symbols: {symbols}")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    return combined
