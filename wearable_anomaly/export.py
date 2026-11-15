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
    if bench.exists():
        export_web_benchmark(json.loads(bench.read_text()))


WEB_BENCHMARK_PATH = config.ROOT / "web" / "lib" / "benchmark.json"


def export_web_benchmark(report: dict) -> None:
    """Trimmed benchmark for the frontend (Vercel builds only web/, so it can't read reports/)."""
    rows = []
    for name, r in report["detectors"].items():
        d = r["detection"]
        rows.append(
            {
                "name": name,
                "f1": d["f1"],
                "pr_auc": d["pr_auc"],
                "precision": d["precision"],
                "recall": d["recall"],
                "flagged_pct": d["flagged_pct"],
                "clean_false_alarm_pct": r["clean_false_alarm_pct"],
                "recall_by_type": {k: v["event_recall"] for k, v in d["per_fault_type"].items()},
            }
        )
    rows.sort(key=lambda row: row["pr_auc"], reverse=True)
    keep = ("test_subjects", "test_rows", "injected_events", "injected_reading_pct")
    out = {**{k: report[k] for k in keep}, "detectors": rows}
    WEB_BENCHMARK_PATH.parent.mkdir(parents=True, exist_ok=True)
    WEB_BENCHMARK_PATH.write_text(json.dumps(out, indent=2))
    print(f"Wrote {WEB_BENCHMARK_PATH.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
