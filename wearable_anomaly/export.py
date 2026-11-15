"""Export the served model as plain JSON.

    python -m wearable_anomaly.export

Reads models/rolling_zscore.joblib (run `train --detector rolling_zscore` first) and
writes api/model/rolling_zscore.json, which is committed and loaded by the API.

Why JSON and not the pickle: the rolling z-score is just numbers (thresholds and
scale floors per sensor). JSON is readable, diffable in git, independent of library
versions, and loading it can't execute code the way unpickling an untrusted file can.
"""

import json

import joblib

from wearable_anomaly import config
from wearable_anomaly.train import model_path

SERVED = "rolling_zscore"
EXPORT_PATH = config.ROOT / "api" / "model" / f"{SERVED}.json"


def main() -> None:
    bundle = joblib.load(model_path(SERVED))
    meta = dict(bundle["meta"])
    bench = config.REPORTS_DIR / "benchmark.json"
    if bench.exists():  # carry the benchmark numbers with the model, for /health
        r = json.loads(bench.read_text())["detectors"][SERVED]
        meta["benchmark"] = {
            "f1": r["detection"]["f1"],
            "pr_auc": r["detection"]["pr_auc"],
            "clean_false_alarm_pct": r["clean_false_alarm_pct"],
        }
    EXPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    EXPORT_PATH.write_text(
        json.dumps({"detector": bundle["detector"].to_dict(), "meta": meta}, indent=2)
    )
    print(
        f"Wrote {EXPORT_PATH.relative_to(config.ROOT)} ({EXPORT_PATH.stat().st_size / 1e3:.1f} KB)"
    )


if __name__ == "__main__":
    main()
