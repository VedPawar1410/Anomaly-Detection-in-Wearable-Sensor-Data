"""Train and calibrate a detector.

    python -m wearable_anomaly.train                       # default: per_sensor_rf
    python -m wearable_anomaly.train --detector <name>     # any name in detectors.REGISTRY

Writes models/<name>.joblib (git-ignored; rebuild it with this command).
"""

import argparse
import time
from datetime import date

import joblib
import sklearn

from wearable_anomaly import config, detectors
from wearable_anomaly.data import load_subjects
from wearable_anomaly.preprocess import clean, split_by_subject

DEFAULT_DETECTOR = "per_sensor_rf"


def model_path(name: str):
    return config.MODELS_DIR / f"{name}.joblib"


def load_splits() -> dict:
    subjects = config.TRAIN_SUBJECTS + config.CALIB_SUBJECTS + config.TEST_SUBJECTS
    return split_by_subject(clean(load_subjects(subjects)))


def train_detector(name: str, splits: dict) -> dict:
    """Fit on the training subjects, calibrate on the calibration subject, save, return the bundle."""
    detector = detectors.make(name)
    # Row-wise models train on every Nth row (compute budget, see config). Window-based
    # detectors need contiguous samples, so they set train_stride = 1 and subsample internally.
    stride = getattr(detector, "train_stride", config.TRAIN_STRIDE)
    train = splits["train"].iloc[::stride]
    print(
        f"[{name}] training on {len(train):,} rows (stride {stride}), subjects {config.TRAIN_SUBJECTS}"
    )

    t0 = time.time()
    detector.fit(train)
    fit_s = time.time() - t0
    detector.calibrate(splits["calib"])
    print(f"[{name}] fitted in {fit_s:.0f} s, calibrated on subject {config.CALIB_SUBJECTS}")

    bundle = {
        "detector": detector,
        "meta": {
            "trained_on": str(date.today()),
            "sklearn_version": sklearn.__version__,
            "train_subjects": config.TRAIN_SUBJECTS,
            "calib_subjects": config.CALIB_SUBJECTS,
            "train_rows": len(train),
            "train_stride": stride,
            "fit_seconds": round(fit_s, 1),
            "params": repr(getattr(detector, "estimator", detector)),
        },
    }
    config.MODELS_DIR.mkdir(exist_ok=True)
    path = model_path(name)
    joblib.dump(bundle, path, compress=3)
    print(f"[{name}] saved {path.relative_to(config.ROOT)} ({path.stat().st_size / 1e6:.1f} MB)")
    return bundle


def load_or_train(name: str, splits: dict, retrain: bool = False) -> dict:
    path = model_path(name)
    if path.exists() and not retrain:
        return joblib.load(path)
    return train_detector(name, splits)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", default=DEFAULT_DETECTOR, choices=sorted(detectors.REGISTRY))
    args = parser.parse_args()
    train_detector(args.detector, load_splits())


if __name__ == "__main__":
    main()
