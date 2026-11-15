"""Train and calibrate the per-sensor detector.

    python -m wearable_anomaly.train

Writes models/per_sensor_rf.joblib (git-ignored; rebuild it with this command).
"""

import time
from datetime import date

import joblib
import sklearn

from wearable_anomaly import config
from wearable_anomaly.data import load_subjects
from wearable_anomaly.detectors.per_sensor import PerSensorRegressor
from wearable_anomaly.preprocess import clean, split_by_subject

MODEL_PATH = config.MODELS_DIR / "per_sensor_rf.joblib"


def load_splits() -> dict:
    subjects = config.TRAIN_SUBJECTS + config.CALIB_SUBJECTS + config.TEST_SUBJECTS
    return split_by_subject(clean(load_subjects(subjects)))


def main() -> None:
    splits = load_splits()
    train = splits["train"].iloc[:: config.TRAIN_STRIDE]
    print(
        f"Training on {len(train):,} rows (every {config.TRAIN_STRIDE}th of "
        f"{len(splits['train']):,}), subjects {config.TRAIN_SUBJECTS}"
    )

    t0 = time.time()
    detector = PerSensorRegressor().fit(train)
    fit_s = time.time() - t0
    print(f"Fitted {len(detector.models_)} models in {fit_s:.0f} s")

    detector.calibrate(splits["calib"])
    print(
        f"Calibrated thresholds on subject {config.CALIB_SUBJECTS} "
        f"({len(splits['calib']):,} rows, target false-alarm rate "
        f"{config.TARGET_FALSE_ALARM_RATE:.1%})"
    )

    config.MODELS_DIR.mkdir(exist_ok=True)
    bundle = {
        "detector": detector,
        "meta": {
            "trained_on": str(date.today()),
            "sklearn_version": sklearn.__version__,
            "train_subjects": config.TRAIN_SUBJECTS,
            "calib_subjects": config.CALIB_SUBJECTS,
            "train_rows": len(train),
            "train_stride": config.TRAIN_STRIDE,
            "fit_seconds": round(fit_s, 1),
            "rf_params": {k: v for k, v in config.RF_PARAMS.items() if k != "n_jobs"},
        },
    }
    joblib.dump(bundle, MODEL_PATH, compress=3)
    print(f"Saved {MODEL_PATH.relative_to(config.ROOT)} ({MODEL_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
