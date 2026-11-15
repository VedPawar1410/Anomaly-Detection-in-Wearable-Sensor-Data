"""Rolling z-score: each sensor against its own recent past.

No model, no learning beyond per-sensor scales: the simplest thing that could work,
and the bar every learned detector has to beat.

Signals per reading, each with its own calibrated threshold:
- jump:  |x - median of the previous 0.5 s| / std of the previous 5 s
- stuck: how many identical values in a row (frozen sensors and dropouts repeat exactly)
- level (RollingZScoreDrift only): |median of the last 0.5 s - median of a baseline
         that ended 5 s ago| / std of the previous 30 s. A slow drift never looks like a
         jump against the last 0.5 s, but it does move away from where the sensor was.
The score is the largest signal relative to its threshold. The false-alarm budget is
split evenly between the signals, so adding one doesn't raise the target.
Everything is causal (past only), so the same detector runs on a live stream.
"""

import numpy as np
import pandas as pd

from wearable_anomaly import config
from wearable_anomaly.features import past_rolling, run_length, subject_blocks


class RollingZScore:
    name = "rolling_zscore"
    train_stride = 1  # fit only needs per-sensor scales

    def __init__(
        self,
        columns=None,
        center_window=50,
        scale_window=500,
        false_alarm_rate=None,
        level_lag=None,
        level_window=None,
        level_scale_window=None,
    ):
        self.columns = list(columns or config.SENSOR_COLUMNS)
        self.center_window = center_window
        self.scale_window = scale_window
        self.false_alarm_rate = false_alarm_rate or config.TARGET_FALSE_ALARM_RATE
        self.level_lag = level_lag  # None: no level signal
        self.level_window = level_window
        self.level_scale_window = level_scale_window
        self.scale_floor_ = None
        self.thresholds_ = None  # (jump, stuck) or (jump, stuck, level), one value per sensor

    @property
    def uses_level(self) -> bool:
        # getattr: models pickled before the level signal existed have no level_lag at all
        return getattr(self, "level_lag", None) is not None

    # --- portable JSON form: the served model is plain numbers, no pickle ------
    def to_dict(self) -> dict:
        tj, ts, *tl = self.thresholds_
        d = {
            "name": self.name,
            "columns": self.columns,
            "center_window": self.center_window,
            "scale_window": self.scale_window,
            "false_alarm_rate": self.false_alarm_rate,
            "scale_floor": [float(v) for v in self.scale_floor_],
            "jump_threshold": [float(v) for v in tj],
            "stuck_threshold": [float(v) for v in ts],
        }
        if self.uses_level:
            d.update(
                level_lag=self.level_lag,
                level_window=self.level_window,
                level_scale_window=self.level_scale_window,
                level_threshold=[float(v) for v in tl[0]],
            )
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RollingZScore":
        """Rebuild either variant from its JSON (older files without `name` load as the base)."""
        klass = RollingZScoreDrift if d.get("level_lag") else RollingZScore
        det = klass(d["columns"], d["center_window"], d["scale_window"], d["false_alarm_rate"])
        det.scale_floor_ = np.asarray(d["scale_floor"])
        thresholds = [np.asarray(d["jump_threshold"]), np.asarray(d["stuck_threshold"])]
        if d.get("level_lag"):
            det.level_lag = d["level_lag"]
            det.level_window = d["level_window"]
            det.level_scale_window = d["level_scale_window"]
            thresholds.append(np.asarray(d["level_threshold"]))
        det.thresholds_ = tuple(thresholds)
        return det

    def fit(self, train: pd.DataFrame) -> "RollingZScore":
        # Floor for the rolling std, so a perfectly calm stretch doesn't divide by ~0.
        self.scale_floor_ = (0.01 * train[self.columns].std()).to_numpy() + 1e-9
        return self

    def _raw(self, df: pd.DataFrame) -> list[np.ndarray]:
        n_signals = 3 if self.uses_level else 2
        out = [np.zeros((len(df), len(self.columns)), dtype=np.float32) for _ in range(n_signals)]
        for rows, block in subject_blocks(df):
            X = block[self.columns]
            center = past_rolling(X, self.center_window, "median", min_periods=5)
            scale = past_rolling(X, self.scale_window, "std", min_periods=20)
            jump = (X - center).abs() / np.maximum(scale, self.scale_floor_)
            out[0][rows] = jump.fillna(0).to_numpy()
            out[1][rows] = run_length(X.to_numpy())
            if self.uses_level:
                recent = X.rolling(self.center_window, min_periods=5).median()
                baseline = (
                    X.shift(self.level_lag)
                    .rolling(self.level_window, min_periods=self.level_window // 4)
                    .median()
                )
                long_scale = past_rolling(X, self.level_scale_window, "std", min_periods=200)
                level = (recent - baseline).abs() / np.maximum(long_scale, self.scale_floor_)
                out[2][rows] = level.fillna(0).to_numpy()
        return out

    def calibrate(self, calib: pd.DataFrame) -> "RollingZScore":
        signals = self._raw(calib)
        q = 1 - self.false_alarm_rate / len(signals)  # split the budget across signals
        thresholds = [np.quantile(s, q, axis=0) for s in signals]
        thresholds[1] = np.maximum(thresholds[1], 2)  # stuck needs at least 2 repeats
        self.thresholds_ = tuple(thresholds)
        return self

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        signals = self._raw(df)
        best = np.max([s / t for s, t in zip(signals, self.thresholds_, strict=True)], axis=0)
        return pd.DataFrame(best, columns=self.columns, index=df.index)


class RollingZScoreDrift(RollingZScore):
    """The rolling z-score plus the level (drift) signal. Settings were chosen on injected
    faults in training subjects 101/102, never on the test subjects."""

    name = "rolling_zscore_drift"

    def __init__(self, columns=None, center_window=50, scale_window=500, false_alarm_rate=None):
        super().__init__(
            columns,
            center_window,
            scale_window,
            false_alarm_rate,
            level_lag=500,  # baseline ends 5 s ago...
            level_window=1000,  # ...and covers 10 s (5-15 s ago), so a 3-10 s drift stands out
            level_scale_window=3000,  # scaled by the spread over the last 30 s
        )
