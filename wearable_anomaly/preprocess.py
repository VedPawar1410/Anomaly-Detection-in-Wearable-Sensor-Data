"""Turn raw PAMAP2 rows into clean, model-ready sensor data.

Every step works within one subject, because each subject is a separate
recording and values must never be filled across the boundary between two people.
"""

import pandas as pd

from wearable_anomaly import config


def clean(raw: pd.DataFrame, return_stats: bool = False):
    """Apply the cleaning recipe. Returns a DataFrame of META_COLUMNS + SENSOR_COLUMNS.

    1. Drop activity 0 (transient time; the dataset authors say to discard it).
    2. Keep only the sensor columns in config.SENSOR_COLUMNS (drops orientation and acc6).
    3. Heart rate: carry the last reading forward. It's sampled ~9 times a second
       against 100 for the IMUs, so ~91% of raw HR cells are NaN by design.
    4. IMU dropouts: interpolate gaps of up to MAX_INTERP_GAP samples (0.1 s).
    5. Drop rows that still have NaNs (longer dropouts, and HR before its first reading).
    """
    stats = {"raw_rows": len(raw)}
    keep = config.META_COLUMNS + config.SENSOR_COLUMNS
    df = raw.loc[raw["activity_id"] != config.TRANSIENT_ACTIVITY, keep]
    stats["dropped_transient"] = len(raw) - len(df)

    imu_cols = [c for c in config.SENSOR_COLUMNS if c != "heart_rate"]
    parts = []
    for _, g in df.groupby("subject", sort=False):
        g = g.copy()
        g["heart_rate"] = g["heart_rate"].ffill()
        g[imu_cols] = g[imu_cols].interpolate(limit=config.MAX_INTERP_GAP, limit_area="inside")
        parts.append(g)
    df = pd.concat(parts)

    before = len(df)
    df = df.dropna().reset_index(drop=True)
    stats["dropped_unfillable_nan"] = before - len(df)
    stats["clean_rows"] = len(df)
    return (df, stats) if return_stats else df


def split_by_subject(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split cleaned data into train / calib / test by subject (see config for why)."""
    groups = {
        "train": config.TRAIN_SUBJECTS,
        "calib": config.CALIB_SUBJECTS,
        "test": config.TEST_SUBJECTS,
    }
    return {
        name: df[df["subject"].isin(ids)].reset_index(drop=True) for name, ids in groups.items()
    }
