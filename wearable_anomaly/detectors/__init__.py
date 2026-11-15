"""Anomaly detectors.

Every detector has the same interface, so the benchmark can treat them alike:

    fit(train_df)        learn what normal looks like
    calibrate(calib_df)  pick thresholds on a clean, unseen subject
    score(df)            DataFrame (rows x sensors); a score above 1 is an anomaly

A score is "how far past the threshold": 1.0 is exactly at it, for every sensor.
"""

import importlib.util

from wearable_anomaly.detectors.autoencoder import ConvAutoencoder
from wearable_anomaly.detectors.isolation import WindowIsolationForest
from wearable_anomaly.detectors.per_sensor import PerSensorHGB, PerSensorRegressor
from wearable_anomaly.detectors.rolling import RollingZScore, RollingZScoreDrift

# name -> class. Train any of these with: python -m wearable_anomaly.train --detector <name>
REGISTRY = {
    cls.name: cls
    for cls in (
        RollingZScore,
        RollingZScoreDrift,
        PerSensorRegressor,
        PerSensorHGB,
        WindowIsolationForest,
        ConvAutoencoder,
    )
}


def make(name: str):
    return REGISTRY[name]()


def available() -> list[str]:
    """Registered detectors whose optional dependency (e.g. torch) is installed."""
    return sorted(
        name
        for name, cls in REGISTRY.items()
        if importlib.util.find_spec(getattr(cls, "optional_dependency", "numpy")) is not None
    )
