"""Flag anomalous readings in a PAMAP2-format recording.

    python -m wearable_anomaly.detect data/raw/PAMAP2_Dataset/Protocol/subject108.dat -o flags.csv
    python -m wearable_anomaly.detect <file.dat> --detector <name>

Writes one row per flagged reading: subject, timestamp, activity_id, sensor,
value, predicted, score. `predicted` is the model's expected value, which can be
used to replace the reading (the original project's imputation idea) without
overwriting anything in the input.
"""

import argparse
import io
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from wearable_anomaly import config, detectors
from wearable_anomaly.data import read_dat
from wearable_anomaly.preprocess import clean
from wearable_anomaly.train import DEFAULT_DETECTOR, model_path

UNKNOWN_ACTIVITY = -1  # for CSVs without labels; never dropped as "transient" (0)


def parse_upload(content: bytes, filename: str) -> pd.DataFrame:
    """Turn uploaded bytes into a raw DataFrame that `clean()` accepts.

    Accepts a PAMAP2 .dat file (54 columns, whitespace-separated, no header) or a CSV
    whose header includes every column in config.SENSOR_COLUMNS. Raises ValueError
    with a readable message for anything else.
    """
    buf = io.BytesIO(content)
    if filename.lower().endswith(".csv"):
        df = pd.read_csv(buf)
        missing = [c for c in config.SENSOR_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"CSV is missing {len(missing)} sensor columns, e.g. {missing[:3]}")
        df[config.SENSOR_COLUMNS] = df[config.SENSOR_COLUMNS].astype(np.float32)
        if "timestamp" not in df:
            df["timestamp"] = np.arange(len(df), dtype=np.float32) / config.SAMPLE_RATE_HZ
        if "activity_id" not in df:
            df["activity_id"] = UNKNOWN_ACTIVITY
    else:
        try:
            df = read_dat(buf)
        except ValueError as e:
            raise ValueError(f"not a PAMAP2 .dat file (54 numeric columns): {e}") from e
    match = re.search(r"(\d+)", Path(filename).stem)
    df.insert(0, "subject", int(match.group(1)) if match else 0)
    return df


def detect_frame(df: pd.DataFrame, detector) -> pd.DataFrame:
    """Return flagged readings in long format for a cleaned DataFrame.

    `predicted` is filled only for detectors that predict values (e.g. per_sensor_rf).
    """
    scores = detector.score(df).to_numpy()
    rows, cols = (scores > 1).nonzero()
    if hasattr(detector, "predict"):
        predicted = detector.predict(df).to_numpy()[rows, cols]
    else:
        predicted = np.full(len(rows), np.nan)
    return pd.DataFrame(
        {
            "subject": df["subject"].to_numpy()[rows],
            "timestamp": df["timestamp"].to_numpy()[rows],
            "activity_id": df["activity_id"].to_numpy()[rows],
            "sensor": [detector.columns[c] for c in cols],
            "value": df[detector.columns].to_numpy()[rows, cols],
            "predicted": predicted,
            "score": scores[rows, cols],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("input", type=Path, help="PAMAP2 .dat file, or a CSV with sensor columns")
    parser.add_argument("-o", "--output", type=Path, default=Path("flags.csv"))
    parser.add_argument("--detector", default=DEFAULT_DETECTOR, choices=sorted(detectors.REGISTRY))
    args = parser.parse_args()

    detector = joblib.load(model_path(args.detector))["detector"]
    df = clean(parse_upload(args.input.read_bytes(), args.input.name))

    flagged = detect_frame(df, detector)
    flagged.to_csv(args.output, index=False)
    total = len(df) * len(detector.columns)
    print(
        f"{len(df):,} rows x {len(detector.columns)} sensors: "
        f"{len(flagged):,} readings flagged ({100 * len(flagged) / total:.2f}%)"
    )
    if len(flagged):
        print("Most-flagged sensors:")
        print(flagged["sensor"].value_counts().head(5).to_string())
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
