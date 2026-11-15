"""Inject labelled sensor faults into clean data.

PAMAP2 has no labelled anomalies, so the only way to measure precision and
recall is to plant faults we know about and check which ones get caught.

Fault types. Magnitudes are in units of each sensor's training standard deviation,
so a "3 std offset" means the same thing for heart rate (bpm) and a gyro (rad/s).

| type    | what happens                                  | real-world cause                 |
|---------|-----------------------------------------------|----------------------------------|
| spike   | one reading jumps by 4-8 std                  | bit error, a knock on the sensor |
| stuck   | repeats its last value for 1-3 s              | frozen firmware / sensor         |
| offset  | constant +-3 std shift for 1-3 s              | calibration jump                 |
| drift   | shift ramps from 0 to +-4 std over 3-10 s     | slow bias / temperature drift    |
| dropout | reads exactly 0 for 0.5-3 s                   | lost wireless packets as zeros   |
| noise   | extra Gaussian noise (2 std) for 1-3 s        | loose sensor, interference       |
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from wearable_anomaly import config

FAULT_TYPES = ("spike", "stuck", "offset", "drift", "dropout", "noise")
WINDOW_TYPES = FAULT_TYPES[1:]


@dataclass
class FaultSpec:
    spikes_per_sensor: int = 20  # per subject
    windows_per_type: int = 2  # per window type, per sensor, per subject
    # (min, max) window length in samples at 100 Hz
    lengths: dict = None
    spike_std: tuple = (4.0, 8.0)
    offset_std: float = 3.0
    drift_std: float = 4.0
    noise_std: float = 2.0

    def __post_init__(self):
        self.lengths = self.lengths or {
            "stuck": (100, 300),
            "offset": (100, 300),
            "drift": (300, 1000),
            "dropout": (50, 300),
            "noise": (100, 300),
        }


def _apply(kind, seg, before, sd, spec, rng):
    """Return the corrupted version of window `seg` (the reading before it is `before`)."""
    n = len(seg)
    sign = rng.choice([-1, 1])
    if kind == "stuck":
        return np.full(n, before)
    if kind == "offset":
        return seg + sign * spec.offset_std * sd
    if kind == "drift":
        return seg + sign * spec.drift_std * sd * np.arange(1, n + 1) / n
    if kind == "dropout":
        return np.zeros(n)
    if kind == "noise":
        return seg + rng.normal(0, spec.noise_std * sd, n)
    raise ValueError(kind)


def inject(
    df: pd.DataFrame,
    train_std: pd.Series,
    columns=None,
    spec: FaultSpec | None = None,
    seed: int = config.RANDOM_SEED,
):
    """Return (corrupted_df, event_ids, events).

    event_ids: int32 array, rows x columns, 0 for clean readings, else the event id.
    events: DataFrame with one row per event (id, type, column, subject, start, length).
    Faults are placed within one subject at a time and never overlap in a sensor.
    """
    columns = list(columns or config.SENSOR_COLUMNS)
    spec = spec or FaultSpec()
    rng = np.random.default_rng(seed)
    values = df[columns].to_numpy(dtype=np.float64, copy=True)
    event_ids = np.zeros(values.shape, dtype=np.int32)
    events = []
    subjects = df["subject"].to_numpy() if "subject" in df else np.zeros(len(df))

    def add(kind, j, start, length):
        events.append(
            {
                "id": len(events) + 1,
                "type": kind,
                "column": columns[j],
                "subject": int(subjects[start]),
                "start": start,
                "length": length,
            }
        )
        event_ids[start : start + length, j] = len(events)

    for subj in np.unique(subjects):
        idx = np.flatnonzero(subjects == subj)
        lo, hi = idx[0], idx[-1] + 1
        for j, col in enumerate(columns):
            sd = float(train_std[col])
            # windows first (they need room), then spikes in the remaining clean readings
            for kind in WINDOW_TYPES:
                for _ in range(spec.windows_per_type):
                    for _attempt in range(50):
                        length = int(rng.integers(*spec.lengths[kind], endpoint=True))
                        start = int(rng.integers(lo + 1, hi - length))
                        if not event_ids[start - 1 : start + length + 1, j].any():
                            break
                    else:
                        continue  # no room left; skip this window
                    seg = values[start : start + length, j]
                    values[start : start + length, j] = _apply(
                        kind, seg, values[start - 1, j], sd, spec, rng
                    )
                    add(kind, j, start, length)
            clean_rows = idx[event_ids[lo:hi, j] == 0]
            for start in rng.choice(clean_rows, size=spec.spikes_per_sensor, replace=False):
                values[start, j] += rng.choice([-1, 1]) * rng.uniform(*spec.spike_std) * sd
                add("spike", j, int(start), 1)

    corrupted = df.copy()
    corrupted[columns] = values.astype(np.float32)
    return corrupted, event_ids, pd.DataFrame(events)
