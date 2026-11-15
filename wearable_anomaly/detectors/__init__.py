"""Anomaly detectors.

Every detector has the same interface, so the benchmark can treat them alike:

    fit(train_df)        learn what normal looks like
    calibrate(calib_df)  pick thresholds on a clean, unseen subject
    score(df)            DataFrame (rows x sensors); a score above 1 is an anomaly

A score is "how far past the threshold": 1.0 is exactly at it, for every sensor.
"""

from wearable_anomaly.detectors.per_sensor import PerSensorRegressor

# name -> class. Train any of these with: python -m wearable_anomaly.train --detector <name>
REGISTRY = {
    PerSensorRegressor.name: PerSensorRegressor,
}


def make(name: str):
    return REGISTRY[name]()
