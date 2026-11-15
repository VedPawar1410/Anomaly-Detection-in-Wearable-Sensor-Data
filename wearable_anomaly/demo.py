"""Build the demo recording the live stream replays.

    python -m wearable_anomaly.demo

Writes (both committed, ~2 MB together):
    api/demo/demo.csv.gz   20 s of each of 7 activities from test subject 108, with faults injected
    api/demo/faults.json   the injected faults (ground truth), so the UI can show injected vs caught

Built offline because fault magnitudes depend on training-set statistics, which
the server doesn't have. The server replays these exact values.
"""

import json

import pandas as pd

from wearable_anomaly import config, faults
from wearable_anomaly.data import load_subject, load_subjects
from wearable_anomaly.preprocess import clean

DEMO_SUBJECT = 108  # a test subject: no model was trained or calibrated on it
# lying, standing, walking, running, cycling, ascending stairs, rope jumping
DEMO_ACTIVITIES = [1, 3, 4, 5, 6, 12, 24]
SEGMENT_SAMPLES = 20 * config.SAMPLE_RATE_HZ
FAULT_SENSORS = [
    "heart_rate",
    "hand_acc16_x",
    "chest_acc16_z",
    "ankle_acc16_y",
    "hand_gyro_z",
    "ankle_gyro_y",
    "chest_mag_x",
    "hand_temp",
]
DEMO_DIR = config.ROOT / "api" / "demo"
DEMO_CSV = DEMO_DIR / "demo.csv.gz"
DEMO_FAULTS = DEMO_DIR / "faults.json"


def main() -> None:
    df = clean(load_subject(DEMO_SUBJECT))
    segments = [df[df.activity_id == a].iloc[:SEGMENT_SAMPLES] for a in DEMO_ACTIVITIES]
    demo = pd.concat(segments, ignore_index=True)

    train_std = clean(load_subjects(config.TRAIN_SUBJECTS))[FAULT_SENSORS].std()
    spec = faults.FaultSpec(
        spikes_per_sensor=3,
        windows_per_type=1,
        lengths={k: (100, 300) for k in faults.WINDOW_TYPES},
    )
    corrupted, _, events = faults.inject(demo, train_std, FAULT_SENSORS, spec)

    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["timestamp", "activity_id"] + config.SENSOR_COLUMNS
    # %.7g keeps full float32 precision, so exact repeats (stuck faults) stay exact
    corrupted[cols].to_csv(DEMO_CSV, index=False, float_format="%.7g")
    events = events.rename(columns={"column": "sensor"})[["type", "sensor", "start", "length"]]
    DEMO_FAULTS.write_text(json.dumps(events.to_dict(orient="records"), indent=1))
    print(
        f"Wrote {DEMO_CSV.relative_to(config.ROOT)} ({len(demo):,} rows, "
        f"{DEMO_CSV.stat().st_size / 1e6:.1f} MB) and {len(events)} faults to "
        f"{DEMO_FAULTS.relative_to(config.ROOT)}"
    )


if __name__ == "__main__":
    main()
