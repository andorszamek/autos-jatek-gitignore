"""Model signal -> target position."""
# Stub — implemented in Phase 6
import pandas as pd
from typing import Any


def decide(
    latest_features: pd.DataFrame,
    current_position: float,
    model: object,
    cfg: dict[str, Any],
) -> tuple[str, float]:
    """Return (side, qty) based on model probability."""
    raise NotImplementedError("Implemented in Phase 6")
