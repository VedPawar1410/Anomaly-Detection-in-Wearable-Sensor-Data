"""Per-sensor regression detector: the 2024 project's idea, with its bugs fixed.

For each sensor, a regressor predicts that sensor from all the other sensors at
the same moment. A reading is anomalous when it's far from its prediction.

What changed from 2024:
- Residuals use abs() on both sides. The old rule `diff > 0.1 * actual` flagged
  every negative reading and almost every reading near zero.
- Each sensor gets its own threshold, set on a clean unseen subject at a chosen
  false-alarm rate, instead of a hand-picked "10% of the value".
- Scores are residual / threshold, so 1.0 means "at the threshold" for every
  sensor and scores are comparable across sensors with different units.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor

from wearable_anomaly import config


class PerSensorRegressor:
    name = "per_sensor_rf"

    def __init__(self, columns=None, estimator=None, false_alarm_rate=None):
        self.columns = list(columns or config.SENSOR_COLUMNS)
        self.estimator = estimator or RandomForestRegressor(**config.RF_PARAMS)
        self.false_alarm_rate = false_alarm_rate or config.TARGET_FALSE_ALARM_RATE
        self.models_: list = []
        self.thresholds_: np.ndarray | None = None

    def _xy(self, df: pd.DataFrame, j: int):
        X = df[self.columns].to_numpy(dtype=np.float32)
        return np.delete(X, j, axis=1), X[:, j]

    def fit(self, train: pd.DataFrame) -> "PerSensorRegressor":
        self.models_ = []
        for j in range(len(self.columns)):
            X, y = self._xy(train, j)
            self.models_.append(clone(self.estimator).fit(X, y))
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Each sensor's expected value, predicted from the other sensors."""
        X = df[self.columns].to_numpy(dtype=np.float32)
        preds = np.column_stack(
            [m.predict(np.delete(X, j, axis=1)) for j, m in enumerate(self.models_)]
        )
        return pd.DataFrame(preds, columns=self.columns, index=df.index)

    def calibrate(self, calib: pd.DataFrame) -> "PerSensorRegressor":
        """Set one threshold per sensor from residuals on clean, unseen data."""
        residuals = (calib[self.columns] - self.predict(calib)).abs()
        self.thresholds_ = residuals.quantile(1 - self.false_alarm_rate).to_numpy()
        return self

    def score(self, df: pd.DataFrame, preds: pd.DataFrame | None = None) -> pd.DataFrame:
        """Residual divided by the sensor's threshold. Above 1 means anomaly."""
        if self.thresholds_ is None:
            raise RuntimeError("call calibrate() before score()")
        preds = self.predict(df) if preds is None else preds
        return (df[self.columns] - preds).abs() / self.thresholds_

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.score(df) > 1
