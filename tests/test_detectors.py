import numpy as np
import pandas as pd
import pytest

from wearable_anomaly import detectors
from wearable_anomaly.detectors.isolation import WindowIsolationForest
from wearable_anomaly.detectors.rolling import RollingZScore
from wearable_anomaly.features import run_length

COLS = ["a", "b"]


def _signal(n, seed):
    """Two smooth, noisy sensors that look like slow body movement."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 100
    df = pd.DataFrame(
        {
            "a": np.sin(2 * np.pi * 0.5 * t) + rng.normal(0, 0.05, n),
            "b": np.cos(2 * np.pi * 0.3 * t) + rng.normal(0, 0.05, n),
        }
    )
    df.insert(0, "subject", 1)
    return df


def _faulty(seed=3):
    df = _signal(5000, seed)
    df.loc[2000, "a"] += 3.0  # spike
    df.loc[3000:3199, "b"] = df.loc[2999, "b"]  # stuck for 2 s
    return df


def test_run_length_counts_identical_values():
    a = np.array([[1.0], [1.0], [2.0], [2.0], [2.0], [3.0]])
    assert run_length(a).ravel().tolist() == [1, 2, 1, 2, 3, 1]


@pytest.mark.parametrize("cls", [RollingZScore, WindowIsolationForest])
def test_window_detectors_catch_spike_and_stuck(cls):
    det = cls(COLS, false_alarm_rate=0.01).fit(_signal(20000, 0)).calibrate(_signal(20000, 1))
    flags = det.score(_faulty()) > 1
    # the spike, caught within 50 ms. Window features (spread, roughness) can react a
    # sample or two after the spike itself, so the window detector may flag it slightly late.
    assert flags.loc[2000:2005, "a"].any()
    assert flags.loc[3050:3199, "b"].mean() > 0.5  # most of the frozen stretch
    clean = _signal(20000, 2)
    assert (det.score(clean) > 1).to_numpy().mean() < 0.05


def test_window_features_never_cross_subjects():
    a, b = _signal(500, 0), _signal(500, 1)
    b["subject"] = 2
    b["a"] += 100.0  # a huge level change at the subject boundary
    det = RollingZScore(COLS).fit(a).calibrate(a)
    scores = det.score(pd.concat([a, b], ignore_index=True))
    # the first rows of subject 2 have no history yet, so they can't be compared with subject 1
    assert scores.loc[500:505, "a"].max() < 1


def test_registry_lists_all_classic_detectors():
    assert {"rolling_zscore", "per_sensor_rf", "per_sensor_hgb", "window_iforest"} <= set(
        detectors.REGISTRY
    )
