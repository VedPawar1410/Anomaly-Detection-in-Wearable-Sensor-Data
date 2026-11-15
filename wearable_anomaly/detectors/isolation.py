"""Isolation Forest on trailing windows, one forest per sensor.

Unsupervised: it never sees a fault. Each reading is described by its trailing
1 s window, and the forest learns which descriptions are common. Rare ones
(easy to isolate with a few random splits) score as anomalous.

Features per reading (all causal, all standardised by the training scale):
- deviation:  x - mean of the previous 1 s
- spread:     std of the previous 1 s
- roughness:  mean |difference| between consecutive samples over the previous 1 s
- stuck:      log(1 + number of identical values in a row)
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import IsolationForest

from wearable_anomaly import config
from wearable_anomaly.features import past_rolling, run_length, subject_blocks


class WindowIsolationForest:
    name = "window_iforest"
    train_stride = 1  # windows need contiguous samples; subsampled after features are built

    def __init__(self, columns=None, window=100, false_alarm_rate=None):
        self.columns = list(columns or config.SENSOR_COLUMNS)
        self.window = window
        self.false_alarm_rate = false_alarm_rate or config.TARGET_FALSE_ALARM_RATE
        self.estimator = IsolationForest(**config.IFOREST_PARAMS)
        self.scale_ = None
        self.models_ = []
        self.thresholds_ = None

    def _features(self, df: pd.DataFrame) -> np.ndarray:
        """rows x sensors x 4 features."""
        out = np.zeros((len(df), len(self.columns), 4), dtype=np.float32)
        w, mp = self.window, 10
        for rows, block in subject_blocks(df):
            X = block[self.columns]
            mean = past_rolling(X, w, "mean", mp)
            out[rows, :, 0] = ((X - mean) / self.scale_).fillna(0).to_numpy()
            out[rows, :, 1] = (past_rolling(X, w, "std", mp) / self.scale_).fillna(0).to_numpy()
            rough = X.diff().abs().shift(1).rolling(w, min_periods=mp).mean() / self.scale_
            out[rows, :, 2] = rough.fillna(0).to_numpy()
            out[rows, :, 3] = np.log1p(run_length(X.to_numpy()))
        return out

    def fit(self, train: pd.DataFrame) -> "WindowIsolationForest":
        self.scale_ = train[self.columns].std().to_numpy() + 1e-9
        feats = self._features(train)[:: config.TRAIN_STRIDE]  # compute budget, see config
        self.models_ = [clone(self.estimator).fit(feats[:, j, :]) for j in range(len(self.columns))]
        return self

    def _raw(self, df: pd.DataFrame) -> np.ndarray:
        feats = self._features(df)
        # score_samples is higher for normal points; negate so higher = more anomalous
        return np.column_stack(
            [-m.score_samples(feats[:, j, :]) for j, m in enumerate(self.models_)]
        )

    def calibrate(self, calib: pd.DataFrame) -> "WindowIsolationForest":
        self.thresholds_ = np.quantile(self._raw(calib), 1 - self.false_alarm_rate, axis=0)
        return self

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self._raw(df) / self.thresholds_, columns=self.columns, index=df.index)
