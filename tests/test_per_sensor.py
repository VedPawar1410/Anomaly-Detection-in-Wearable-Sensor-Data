import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from wearable_anomaly.detectors.per_sensor import PerSensorRegressor


def _linked(n, seed):
    """Three sensors that predict each other: b = 2a, c = a + b, plus small noise."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = 2 * a + rng.normal(scale=0.05, size=n)
    c = a + b + rng.normal(scale=0.05, size=n)
    return pd.DataFrame({"a": a, "b": b, "c": c})


def _detector():
    return PerSensorRegressor(["a", "b", "c"], LinearRegression(), false_alarm_rate=0.01)


def test_calibrated_false_alarm_rate_carries_over_to_new_data():
    det = _detector().fit(_linked(2000, 0)).calibrate(_linked(5000, 1))
    rate = det.detect(_linked(5000, 2)).to_numpy().mean()
    assert 0.003 < rate < 0.03


def test_flags_an_injected_spike_including_negative_values():
    det = _detector().fit(_linked(2000, 0)).calibrate(_linked(5000, 1))
    test = _linked(500, 3)
    test.loc[100, "b"] -= 5.0  # a spike downwards
    flags = det.detect(test)
    assert flags.loc[100, "b"]
    # the 2024 bug flagged every negative reading; here negatives are not special
    negative_clean = (test["b"] < 0) & (test.index != 100)
    assert flags.loc[negative_clean, "b"].mean() < 0.05


def test_scores_are_relative_to_each_sensors_threshold():
    det = _detector().fit(_linked(2000, 0)).calibrate(_linked(5000, 1))
    scores = det.score(_linked(500, 4))
    assert list(scores.columns) == ["a", "b", "c"]
    assert (scores >= 0).all().all()
