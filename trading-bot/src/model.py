"""Model training, persistence, and inference."""
# Stub — implemented in Phase 4
import pandas as pd


def train(X: pd.DataFrame, y: pd.Series) -> object:
    raise NotImplementedError("Implemented in Phase 4")


def save(model: object, path: str) -> None:
    raise NotImplementedError("Implemented in Phase 4")


def load(path: str) -> object:
    raise NotImplementedError("Implemented in Phase 4")


def predict_proba(model: object, X: pd.DataFrame) -> pd.Series:
    raise NotImplementedError("Implemented in Phase 4")
