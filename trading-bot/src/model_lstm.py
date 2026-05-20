"""LSTM binary classifier for directional trading signals.

Same interface as model.py: train(), save(), load(), predict_proba().
M3 Mac uses MPS acceleration automatically.

Architecture:
    Input  (batch, SEQ_LEN=60, n_features)
    LSTM   hidden=128, layers=2, dropout=0.2
    Head   Linear(128→64) → ReLU → Dropout(0.3) → Linear(64→1) → Sigmoid
"""
import copy
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SEQ_LEN = 60  # days of lookback per sequence


def _device():
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_net(n_features: int, hidden: int = 128, n_layers: int = 2, dropout: float = 0.2):
    import torch.nn as nn

    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                n_features, hidden, n_layers,
                dropout=dropout if n_layers > 1 else 0.0,
                batch_first=True,
            )
            self.head = nn.Sequential(
                nn.Linear(hidden, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            # x: (batch, seq_len, features)
            _, (h, _) = self.lstm(x)
            return self.head(h[-1]).squeeze(-1)  # last layer hidden state

    return _Net()


class LSTMModel:
    """Wrapper: PyTorch net + StandardScaler."""

    def __init__(self, net, scaler, seq_len: int, n_features: int, device):
        self.net = net
        self.scaler = scaler
        self.seq_len = seq_len
        self.n_features = n_features
        self.device = device


def _make_sequences(X_arr: np.ndarray, y_arr: np.ndarray | None, seq_len: int):
    """Sliding window over time-ordered feature matrix.

    sequence k → X[k : k+seq_len], label = y[k+seq_len-1]
    Returns xs (n_seqs, seq_len, n_features) and ys (n_seqs,) or None.
    """
    n = len(X_arr)
    n_seqs = n - seq_len + 1
    if n_seqs <= 0:
        empty = np.empty((0, seq_len, X_arr.shape[1]), dtype=np.float32)
        return empty, (np.empty(0, dtype=np.float32) if y_arr is not None else None)

    xs = np.stack([X_arr[k: k + seq_len] for k in range(n_seqs)]).astype(np.float32)
    ys = y_arr[seq_len - 1:].astype(np.float32) if y_arr is not None else None
    return xs, ys


def train(X: pd.DataFrame, y: pd.Series, cfg: dict[str, Any] | None = None) -> LSTMModel:
    """Train LSTM binary classifier.

    Time-ordered 80/20 split. StandardScaler fitted on train only.
    Early stopping on val AUC (patience=15).

    Returns:
        LSTMModel wrapper (net + scaler)
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    try:
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ImportError("scikit-learn required: pip install scikit-learn") from exc

    if len(X) < SEQ_LEN + 20:
        raise ValueError(f"[lstm] Not enough samples: {len(X)} < {SEQ_LEN + 20}")

    device = _device()
    n_features = X.shape[1]

    # 80/20 time-ordered split
    split = int(len(X) * 0.80)
    X_arr = X.values.astype(np.float32)
    y_arr = y.values.astype(np.float32)

    # Fit scaler on train only, apply to both
    scaler = StandardScaler()
    X_scaled = X_arr.copy()
    X_scaled[:split] = scaler.fit_transform(X_arr[:split])
    X_scaled[split:] = scaler.transform(X_arr[split:])

    # Create sequences; split by label position (k + SEQ_LEN - 1 < split → train)
    xs_all, ys_all = _make_sequences(X_scaled, y_arr, SEQ_LEN)
    split_seq = max(0, split - SEQ_LEN + 1)
    train_xs, train_ys = xs_all[:split_seq], ys_all[:split_seq]
    val_xs, val_ys = xs_all[split_seq:], ys_all[split_seq:]

    print(f"\n{'=' * 55}")
    print(f"  LSTM Training  (device={device})")
    print(f"  Train sequences : {len(train_xs):>6d}")
    print(f"  Val sequences   : {len(val_xs):>6d}")
    print(f"  Features        : {n_features:>6d}  |  seq_len={SEQ_LEN}")
    print(f"  Target up%%      : {train_ys.mean() * 100:>5.1f}%% train / {val_ys.mean() * 100:.1f}%% val")
    print(f"{'=' * 55}")

    net = _build_net(n_features).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5, min_lr=1e-5
    )
    criterion = nn.BCELoss()

    train_ds = TensorDataset(
        torch.tensor(train_xs, device=device),
        torch.tensor(train_ys, device=device),
    )
    loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_x_t = torch.tensor(val_xs, device=device)

    best_auc = 0.0
    best_state = None
    patience_ctr = 0
    patience = 15

    for epoch in range(100):
        net.train()
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(net(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        net.eval()
        with torch.no_grad():
            val_proba = net(val_x_t).cpu().numpy()

        try:
            val_auc = float(roc_auc_score(val_ys, val_proba))
        except Exception:
            val_auc = 0.5

        scheduler.step(val_auc)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  epoch {epoch + 1:3d}  "
                f"loss={total_loss / len(loader):.4f}  "
                f"val_AUC={val_auc:.4f}  "
                f"lr={lr:.2e}"
            )

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = copy.deepcopy(net.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"  Early stopping at epoch {epoch + 1} (best val_AUC={best_auc:.4f})")
                break

    net.load_state_dict(best_state)
    net.eval()

    val_pred_bin = (val_proba >= 0.5).astype(int)
    val_acc = float((val_pred_bin == val_ys.astype(int)).mean())

    print(f"\n{'=' * 55}")
    print(f"  Training complete — best val AUC : {best_auc:.4f}")
    print(f"  Val Accuracy                     : {val_acc:.4f}  ({val_acc * 100:.1f}%%)")
    print(f"{'=' * 55}\n")

    logger.info("[lstm] Val AUC=%.4f | Val Accuracy=%.4f | device=%s", best_auc, val_acc, device)
    return LSTMModel(net, scaler, SEQ_LEN, n_features, device)


def save(model: LSTMModel, output_dir: str) -> None:
    """Save model to <output_dir>/<timestamp>_model.lstm and copy to latest.lstm."""
    import torch

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    versioned = out / f"{ts}_model.lstm"
    latest = out / "latest.lstm"

    torch.save(
        {
            "state_dict": model.net.state_dict(),
            "scaler": model.scaler,
            "seq_len": model.seq_len,
            "n_features": model.n_features,
        },
        str(versioned),
    )
    shutil.copy2(str(versioned), str(latest))

    logger.info("[lstm] Saved to %s (also copied to %s)", versioned, latest)
    print(f"[lstm] Saved to {versioned}")


def load(path: str) -> LSTMModel:
    """Load LSTMModel from .lstm file."""
    import torch

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"[lstm] Model not found: {path}")

    ckpt = torch.load(str(p), map_location="cpu", weights_only=False)
    net = _build_net(ckpt["n_features"])
    net.load_state_dict(ckpt["state_dict"])
    device = _device()
    net = net.to(device).eval()

    logger.info("[lstm] Loaded from %s (device=%s)", p, device)
    return LSTMModel(net, ckpt["scaler"], ckpt["seq_len"], ckpt["n_features"], device)


def predict_proba(model: LSTMModel, X: pd.DataFrame) -> np.ndarray:
    """Return probability of class 1 (up) for each valid sequence in X.

    X must be time-ordered with at least model.seq_len rows.
    Returns array of shape (max(0, len(X) - seq_len + 1),).
    If not enough rows, returns [0.5] (neutral).
    """
    import torch

    X_arr = model.scaler.transform(X.values.astype(np.float32))
    xs, _ = _make_sequences(X_arr, None, model.seq_len)

    if len(xs) == 0:
        return np.array([0.5], dtype=float)

    model.net.eval()
    with torch.no_grad():
        proba = model.net(torch.tensor(xs, device=model.device)).cpu().numpy()

    if str(model.device) == "mps":
        torch.mps.empty_cache()

    return np.asarray(proba, dtype=float)
