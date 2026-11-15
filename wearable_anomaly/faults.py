"""Inject labelled sensor faults into clean data.

PAMAP2 has no labelled anomalies, so the only way to measure precision and
recall is to plant faults we know about and check which ones get caught.

Fault types (magnitudes are in units of each sensor's training standard deviation):
- spike:  one reading jumps by 4-8 std (e.g. a bit error or a knock on the sensor)
- stuck:  the sensor repeats its last value for 1-3 s (a frozen sensor)
- offset: a constant +-3 std shift for 1-3 s (a calibration jump)
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from wearable_anomaly import config

FAULT_TYPES = ("spike", "stuck", "offset")


@dataclass
class FaultSpec:
    spikes_per_sensor: int = 20  # per subject
    windows_per_type: int = 2  # stuck and offset windows per sensor per subject
    min_len: int = 100  # samples (1 s at 100 Hz)
    max_len: int = 300  # samples (3 s)
    spike_std: tuple = (4.0, 8.0)
    offset_std: float = 3.0


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
            for kind in ("stuck", "offset"):
                for _ in range(spec.windows_per_type):
                    for _attempt in range(20):
                        length = int(rng.integers(spec.min_len, spec.max_len + 1))
                        start = int(rng.integers(lo + 1, hi - length))
                        if not event_ids[start - 1 : start + length + 1, j].any():
                            break
                    else:
                        continue
                    if kind == "stuck":
                        values[start : start + length, j] = values[start - 1, j]
                    else:
                        shift = rng.choice([-1, 1]) * spec.offset_std * sd
                        values[start : start + length, j] += shift
                    add(kind, j, start, length)
            clean_rows = idx[event_ids[lo:hi, j] == 0]
            for start in rng.choice(clean_rows, size=spec.spikes_per_sensor, replace=False):
                values[start, j] += rng.choice([-1, 1]) * rng.uniform(*spec.spike_std) * sd
                add("spike", j, int(start), 1)

    corrupted = df.copy()
    corrupted[columns] = values.astype(np.float32)
    return corrupted, event_ids, pd.DataFrame(events)
