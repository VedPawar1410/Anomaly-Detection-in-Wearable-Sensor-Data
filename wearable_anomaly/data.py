"""Load PAMAP2 protocol files into DataFrames.

    python -m wearable_anomaly.data      # load all subjects, print a summary

Parsing the 1.5 GB of text takes a while, so each subject is cached as a pickle
in data/interim/ after the first load.
"""

import numpy as np
import pandas as pd

from wearable_anomaly import config


def read_dat(path, expected_rows: int | None = None) -> pd.DataFrame:
    """Parse one whitespace-separated PAMAP2 .dat file (54 columns, no header).

    `path` can be a file path or a file-like object (an upload). Raises ValueError
    if the row count differs from `expected_rows`: never skip or lose rows silently.
    """
    name = getattr(path, "name", "upload")
    df = pd.read_csv(path, sep=r"\s+", header=None, names=config.RAW_COLUMNS, dtype=np.float32)
    if expected_rows is not None and len(df) != expected_rows:
        raise ValueError(f"{name}: expected {expected_rows:,} rows, got {len(df):,}")
    df["activity_id"] = df["activity_id"].astype(np.int16)
    return df


def load_subject(subject: int, use_cache: bool = True) -> pd.DataFrame:
    """Load one protocol subject with a `subject` column added."""
    cache = config.CACHE_DIR / f"subject{subject}.pkl"
    if use_cache and cache.exists():
        return pd.read_pickle(cache)

    path = config.PROTOCOL_DIR / f"subject{subject}.dat"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python scripts/download_data.py")
    df = read_dat(path, expected_rows=config.EXPECTED_ROWS[subject])
    df.insert(0, "subject", np.int16(subject))

    if use_cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(cache)
    return df


def load_subjects(subjects: list[int], use_cache: bool = True) -> pd.DataFrame:
    """Load several subjects into one DataFrame, in subject then time order."""
    return pd.concat([load_subject(s, use_cache) for s in subjects], ignore_index=True)


def main() -> None:
    from wearable_anomaly.preprocess import clean

    subjects = sorted(config.EXPECTED_ROWS)
    raw = load_subjects(subjects)
    cleaned, stats = clean(raw, return_stats=True)
    print(f"Raw rows: {len(raw):,} from {len(subjects)} subjects")
    for key, value in stats.items():
        print(f"  {key}: {value:,}")
    print(f"Clean rows: {len(cleaned):,} x {len(config.SENSOR_COLUMNS)} sensor columns")
    print("\nRows per activity after cleaning:")
    counts = cleaned["activity_id"].value_counts().sort_index()
    for act, n in counts.items():
        print(f"  {act:>2} {config.ACTIVITIES[act]:<18} {n:>9,}")


if __name__ == "__main__":
    main()
