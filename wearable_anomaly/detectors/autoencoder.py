"""1D convolutional autoencoder over all sensors at once (benchmark only).

The model squeezes a 1 s window (100 samples x 31 sensors) through a small
bottleneck and rebuilds it. Trained only on normal data, it learns to rebuild
normal movement; a fault it has never seen (a frozen line, a ramp, zeros) gets
rebuilt as normal movement, and the reconstruction error is the anomaly score.

Unlike the rolling z-score, convolutions mix time and sensors, so the model knows
what the other sensors are doing in that same second.

Trade-offs, stated plainly:
- Offline only: a window contains up to 1 s of future samples, so this detector is
  not used in the live stream (Step 9 uses causal detectors).
- Needs torch (requirements-train.txt). torch is imported inside methods, so the rest
  of the package, and the deployed API, work without it.
- Trained on CPU with fixed seeds for reproducibility.
"""

import numpy as np
import pandas as pd

from wearable_anomaly import config
from wearable_anomaly.features import subject_blocks


def _build(n_channels: int, hidden: int, code: int):
    from torch import nn

    return nn.Sequential(
        # encoder: 100 -> 50 -> 25 samples
        nn.Conv1d(n_channels, hidden, kernel_size=5, stride=2, padding=2),
        nn.ReLU(),
        nn.Conv1d(hidden, code, kernel_size=5, stride=2, padding=2),
        nn.ReLU(),
        # decoder: 25 -> 50 -> 100 samples
        nn.ConvTranspose1d(code, hidden, kernel_size=4, stride=2, padding=1),
        nn.ReLU(),
        nn.ConvTranspose1d(hidden, n_channels, kernel_size=4, stride=2, padding=1),
    )


class ConvAutoencoder:
    name = "conv_autoencoder"
    train_stride = 1  # windows need contiguous samples
    optional_dependency = "torch"

    def __init__(self, columns=None, false_alarm_rate=None, **params):
        self.columns = list(columns or config.SENSOR_COLUMNS)
        self.false_alarm_rate = false_alarm_rate or config.TARGET_FALSE_ALARM_RATE
        self.params = {**config.AE_PARAMS, **params}
        self.mean_ = self.std_ = None
        self.model_ = None
        self.thresholds_ = None
        self.train_losses_ = []

    # --- helpers -------------------------------------------------------------
    def _standardise(self, df: pd.DataFrame) -> np.ndarray:
        return ((df[self.columns].to_numpy(np.float32) - self.mean_) / self.std_).astype(np.float32)

    def _window_starts(self, df: pd.DataFrame, step: int) -> np.ndarray:
        """Window starts that never cross a subject boundary; the last window ends at the block end."""
        w, starts = self.params["window"], []
        for rows, _ in subject_blocks(df):
            lo, hi = rows.start, rows.stop
            if hi - lo < w:
                continue
            s = list(range(lo, hi - w + 1, step))
            if s[-1] != hi - w:
                s.append(hi - w)
            starts.extend(s)
        return np.asarray(starts)

    # --- interface -----------------------------------------------------------
    def fit(self, train: pd.DataFrame) -> "ConvAutoencoder":
        import torch

        p = self.params
        torch.manual_seed(config.RANDOM_SEED)
        rng = np.random.default_rng(config.RANDOM_SEED)
        self.mean_ = train[self.columns].mean().to_numpy(np.float32)
        self.std_ = train[self.columns].std().to_numpy(np.float32) + 1e-6
        X = torch.from_numpy(self._standardise(train))
        starts = self._window_starts(train, step=1)
        offsets = torch.arange(p["window"])

        self.model_ = _build(len(self.columns), p["hidden"], p["code"])
        opt = torch.optim.Adam(self.model_.parameters(), lr=p["lr"])
        loss_fn = torch.nn.MSELoss()
        self.model_.train()
        for _ in range(p["epochs"]):
            # a fresh random sample of windows each epoch (compute budget, see config)
            chosen = rng.choice(
                starts, size=min(p["windows_per_epoch"], len(starts)), replace=False
            )
            total = 0.0
            for i in range(0, len(chosen), p["batch_size"]):
                idx = torch.from_numpy(chosen[i : i + p["batch_size"]])[:, None] + offsets
                batch = X[idx].transpose(1, 2)  # (batch, sensors, time)
                opt.zero_grad()
                loss = loss_fn(self.model_(batch), batch)
                loss.backward()
                opt.step()
                total += loss.item() * len(batch)
            self.train_losses_.append(round(total / len(chosen), 5))
        return self

    def _errors(self, df: pd.DataFrame) -> np.ndarray:
        """|x - reconstruction| per reading, in standardised units."""
        import torch

        w = self.params["window"]
        X = self._standardise(df)
        err = np.zeros_like(X)
        self.model_.eval()
        starts = self._window_starts(df, step=w)  # non-overlapping, plus one tail window
        with torch.no_grad():
            for i in range(0, len(starts), 1024):
                s = starts[i : i + 1024]
                idx = s[:, None] + np.arange(w)
                batch = torch.from_numpy(X[idx]).transpose(1, 2)
                recon = self.model_(batch).transpose(1, 2).numpy()
                err[idx] = np.abs(X[idx] - recon)
        return err

    def calibrate(self, calib: pd.DataFrame) -> "ConvAutoencoder":
        self.thresholds_ = np.quantile(self._errors(calib), 1 - self.false_alarm_rate, axis=0)
        return self

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            self._errors(df) / self.thresholds_, columns=self.columns, index=df.index
        )
