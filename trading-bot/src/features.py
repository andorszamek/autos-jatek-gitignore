"""Feature engineering — single source of truth for training and inference."""
# Stub — implemented in Phase 3
import pandas as pd

FEATURE_COLUMNS: list[str] = []


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    raise NotImplementedError("Implemented in Phase 3")
