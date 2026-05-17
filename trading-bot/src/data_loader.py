"""Historical and live market data. Canonical DataFrame format shared by all modules."""
# Stub — implemented in Phase 2
import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

CANONICAL_COLUMNS = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]


def load_history(
    symbols: list[str],
    timeframe: str,
    days: int,
    cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    raise NotImplementedError("Implemented in Phase 2")


def get_latest_bars(
    symbols: list[str],
    lookback: int,
    cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    raise NotImplementedError("Implemented in Phase 2")
