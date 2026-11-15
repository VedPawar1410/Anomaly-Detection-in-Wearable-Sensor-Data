import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from wearable_anomaly import faults
from wearable_anomaly.benchmark import detection_metrics
from wearable_anomaly.detectors.per_sensor import PerSensorRegressor, Rule2024


def _events():
    rows = [
        {"id": i + 1, "type": t, "column": "a", "subject": 1, "start": i, "length": 1}
        for i, t in enumerate(faults.FAULT_TYPES)
    ]
    return pd.DataFrame(rows)


def test_detection_metrics_on_a_hand_checked_case():
    # 10 readings, 1 sensor; readings 0-5 are faults (one of each type), 6-9 clean
    event_ids = np.array([[1], [2], [3], [4], [5], [6], [0], [0], [0], [0]])
    scores = np.array([[2.0], [2.0], [2.0], [0.5], [0.5], [0.5], [2.0], [0.1], [0.1], [0.1]])
    m = detection_metrics(scores, event_ids, _events())
    assert m["precision"] == 0.75  # 3 of 4 flags are faults
    assert m["recall"] == 0.5  # 3 of 6 faults flagged
    assert m["per_fault_type"]["spike"]["event_recall"] == 1.0
    assert m["per_fault_type"]["drift"]["event_recall"] == 0.0
    assert 0 <= m["pr_auc"] <= 1


def test_rule_2024_reproduces_the_negative_value_bug():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=500), "b": rng.normal(size=500)})
    base = PerSensorRegressor(["a", "b"], LinearRegression()).fit(df).calibrate(df)
    flags = Rule2024(base).score(df) > 1
    assert flags.loc[df["a"] < 0, "a"].all()  # every negative reading flagged
