"""Rolling z-score baseline: each sensor against its own recent past.

No model, no learning beyond per-sensor scales: the simplest thing that could work,
and the bar every learned detector has to beat.

Two signals per reading, each with its own calibrated threshold:
- jump:  |x - median of the previous 0.5 s| / std of the previous 5 s
- stuck: how many identical values in a row (frozen sensors and dropouts repeat exactly)
The score is the larger of the two, relative to its threshold.
"""

import numpy as np
import pandas as pd

from wearable_anomaly import config
from wearable_anomaly.features import past_rolling, run_length, subject_blocks


class RollingZScore:
    name = "rolling_zscore"
    train_stride = 1  # fit only needs per-sensor scales

    def __init__(self, columns=None, center_window=50, scale_window=500, false_alarm_rate=None):
        self.columns = list(columns or config.SENSOR_COLUMNS)
        self.center_window = center_window
        self.scale_window = scale_window
        self.false_alarm_rate = false_alarm_rate or config.TARGET_FALSE_ALARM_RATE
        self.scale_floor_ = None
        self.thresholds_ = None

    def fit(self, train: pd.DataFrame) -> "RollingZScore":
        # Floor for the rolling std, so a perfectly calm stretch doesn't divide by ~0.
        self.scale_floor_ = (0.01 * train[self.columns].std()).to_numpy() + 1e-9
        return self

    def _raw(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        jump = np.zeros((len(df), len(self.columns)), dtype=np.float32)
        stuck = np.zeros_like(jump)
        for rows, block in subject_blocks(df):
            X = block[self.columns]
            center = past_rolling(X, self.center_window, "median", min_periods=5)
            scale = past_rolling(X, self.scale_window, "std", min_periods=20)
            z = (X - center).abs() / np.maximum(scale, self.scale_floor_)
            jump[rows] = z.fillna(0).to_numpy()
            stuck[rows] = run_length(X.to_numpy())
        return jump, stuck

    def calibrate(self, calib: pd.DataFrame) -> "RollingZScore":
        jump, stuck = self._raw(calib)
        q = 1 - self.false_alarm_rate / 2  # split the false-alarm budget between the two signals
        self.thresholds_ = (
            np.quantile(jump, q, axis=0),
            np.maximum(np.quantile(stuck, q, axis=0), 2),
        )
        return self

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        jump, stuck = self._raw(df)
        tj, ts = self.thresholds_
        return pd.DataFrame(np.maximum(jump / tj, stuck / ts), columns=self.columns, index=df.index)
