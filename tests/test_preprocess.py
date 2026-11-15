import numpy as np
import pandas as pd

from wearable_anomaly import config
from wearable_anomaly.preprocess import clean, split_by_subject


def _raw(subject, n=30, activity=1):
    df = pd.DataFrame(
        np.ones((n, len(config.RAW_COLUMNS)), dtype=np.float32), columns=config.RAW_COLUMNS
    )
    df["timestamp"] = np.arange(n) / 100
    df["activity_id"] = activity
    df.insert(0, "subject", subject)
    return df


def test_drops_transient_activity_and_unused_columns():
    raw = pd.concat([_raw(101, activity=0), _raw(101, activity=4)], ignore_index=True)
    out = clean(raw)
    assert (out["activity_id"] == 4).all() and len(out) == 30
    assert list(out.columns) == config.META_COLUMNS + config.SENSOR_COLUMNS


def test_heart_rate_forward_fill_never_crosses_subjects():
    a, b = _raw(101), _raw(102)
    a["heart_rate"] = np.nan
    a.loc[0, "heart_rate"] = 80.0
    b["heart_rate"] = np.nan
    b.loc[5, "heart_rate"] = 120.0
    out = clean(pd.concat([a, b], ignore_index=True))
    assert (out.loc[out.subject == 101, "heart_rate"] == 80).all()
    # subject 102's first 5 rows have no earlier reading: dropped, not filled with 80
    s102 = out[out.subject == 102]
    assert len(s102) == 25 and (s102["heart_rate"] == 120).all()


def test_short_gaps_interpolated_long_gaps_dropped():
    raw = _raw(101, n=60)
    raw.loc[10:14, "hand_gyro_x"] = np.nan  # 5 samples: filled
    raw.loc[30:49, "hand_gyro_x"] = np.nan  # 20 samples: too long
    out, stats = clean(raw, return_stats=True)
    assert out["hand_gyro_x"].notna().all()
    assert stats["dropped_unfillable_nan"] == 20 - config.MAX_INTERP_GAP
    assert len(out) == 60 - stats["dropped_unfillable_nan"]


def test_split_by_subject():
    raw = pd.concat([_raw(s) for s in (101, 107, 108, 109)], ignore_index=True)
    parts = split_by_subject(clean(raw))
    assert set(parts["train"].subject) == {101}
    assert set(parts["calib"].subject) == {107}
    assert set(parts["test"].subject) == {108}
