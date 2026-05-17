"""Model training, persistence, and inference.

Uses LightGBM binary classification.
Time-ordered train/val split (no shuffling) to avoid look-ahead bias.
"""
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def train(X: pd.DataFrame, y: pd.Series, cfg: dict[str, Any] | None = None):
    """Train LightGBM binary classifier.

    Uses first 80% of rows for training, last 20% for validation (time-ordered).
    Early stopping on validation AUC (50 rounds).

    Returns:
        lgb.Booster: trained booster
    """
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("lightgbm is required: pip install lightgbm") from exc

    if len(X) < 50:
        raise ValueError(f"[model] Not enough samples to train: {len(X)} < 50")

    split = int(len(X) * 0.80)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]

    logger.info(
        "[model] Training: %d samples, Validation: %d samples",
        len(X_train),
        len(X_val),
    )

    params = {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "num_leaves": 31,
        "learning_rate": 0.05,
        "random_state": 42,
        "verbosity": -1,
        "n_jobs": -1,
    }

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=20),
    ]

    booster = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=200,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=callbacks,
    )

    # Evaluation metrics
    val_proba = booster.predict(X_val)
    val_pred = (val_proba >= 0.5).astype(int)
    y_val_arr = np.array(y_val)

    accuracy = (val_pred == y_val_arr).mean()
    directional_hit = accuracy  # binary classification hit rate

    # AUC from lightgbm eval result
    try:
        from sklearn.metrics import roc_auc_score
        val_auc = roc_auc_score(y_val_arr, val_proba)
    except Exception:
        val_auc = float("nan")

    logger.info(
        "[model] Val AUC=%.4f | Val Accuracy=%.4f | Directional Hit-Rate=%.4f",
        val_auc,
        accuracy,
        directional_hit,
    )
    print(
        f"[model] Val AUC={val_auc:.4f} | Val Accuracy={accuracy:.4f} "
        f"| Directional Hit-Rate={directional_hit:.4f}"
    )

    return booster


def save(model, output_dir: str) -> None:
    """Save model to <output_dir>/<timestamp>_model.lgb and copy to <output_dir>/latest.lgb."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    versioned_path = out / f"{ts}_model.lgb"
    latest_path = out / "latest.lgb"

    model.save_model(str(versioned_path))
    shutil.copy2(str(versioned_path), str(latest_path))

    logger.info("[model] Saved to %s (also copied to %s)", versioned_path, latest_path)
    print(f"[model] Saved to {versioned_path}")


def load(path: str):
    """Load a LightGBM Booster from path."""
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("lightgbm is required: pip install lightgbm") from exc

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"[model] Model file not found: {path}")

    booster = lgb.Booster(model_file=str(p))
    logger.info("[model] Loaded model from %s", path)
    return booster


def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    """Return probability of class 1 (up) for each row in X.

    Returns:
        np.ndarray of shape (n_samples,), values in [0, 1].
    """
    proba = model.predict(X)
    return np.asarray(proba, dtype=float)
