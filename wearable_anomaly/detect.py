"""Flag anomalous readings in a PAMAP2-format recording.

    python -m wearable_anomaly.detect data/raw/PAMAP2_Dataset/Protocol/subject108.dat -o flags.csv

Writes one row per flagged reading: subject, timestamp, activity_id, sensor,
value, predicted, score. `predicted` is the model's expected value, which can be
used to replace the reading (the original project's imputation idea) without
overwriting anything in the input.
"""

import argparse
import re
from pathlib import Path

import joblib
import pandas as pd

from wearable_anomaly.data import read_dat
from wearable_anomaly.preprocess import clean
from wearable_anomaly.train import MODEL_PATH


def detect_frame(df: pd.DataFrame, detector) -> pd.DataFrame:
    """Return flagged readings in long format for a cleaned DataFrame."""
    preds = detector.predict(df)
    scores = detector.score(df, preds)
    flags = scores > 1
    rows, cols = flags.to_numpy().nonzero()
    return pd.DataFrame(
        {
            "subject": df["subject"].to_numpy()[rows],
            "timestamp": df["timestamp"].to_numpy()[rows],
            "activity_id": df["activity_id"].to_numpy()[rows],
            "sensor": [detector.columns[c] for c in cols],
            "value": df[detector.columns].to_numpy()[rows, cols],
            "predicted": preds.to_numpy()[rows, cols],
            "score": scores.to_numpy()[rows, cols],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("input", type=Path, help="PAMAP2 .dat file")
    parser.add_argument("-o", "--output", type=Path, default=Path("flags.csv"))
    args = parser.parse_args()

    detector = joblib.load(MODEL_PATH)["detector"]
    raw = read_dat(args.input)
    match = re.search(r"(\d+)", args.input.stem)
    raw.insert(0, "subject", int(match.group(1)) if match else 0)
    df = clean(raw)

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
