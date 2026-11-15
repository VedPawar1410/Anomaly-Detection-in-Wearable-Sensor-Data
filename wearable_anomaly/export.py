"""Export the served model as plain JSON.

    python -m wearable_anomaly.export

Reads models/<config.SERVED_DETECTOR>.joblib (train it first) and writes
api/model/<config.SERVED_DETECTOR>.json, which is committed and loaded by the API.

Why JSON and not the pickle: the rolling z-score is just numbers (thresholds and
scale floors per sensor). JSON is readable, diffable in git, independent of library
versions, and loading it can't execute code the way unpickling an untrusted file can.
"""

import json

import joblib

from wearable_anomaly import config
from wearable_anomaly.train import model_path

SERVED = config.SERVED_DETECTOR
EXPORT_PATH = config.ROOT / "api" / "model" / f"{SERVED}.json"


def main() -> None:
    bundle = joblib.load(model_path(SERVED))
    meta = dict(bundle["meta"])
    bench = config.REPORTS_DIR / "benchmark.json"
    report = json.loads(bench.read_text()) if bench.exists() else {"detectors": {}}
    r = report["detectors"].get(SERVED)
    if r:  # carry the benchmark numbers with the model, for /health
        meta["benchmark"] = {
            "f1": r["detection"]["f1"],
            "pr_auc": r["detection"]["pr_auc"],
            "clean_false_alarm_pct": r["clean_false_alarm_pct"],
        }
    else:
        print(
            f"Note: {SERVED} isn't in reports/benchmark.json yet; run the benchmark, then export again"
        )
    EXPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    EXPORT_PATH.write_text(
        json.dumps({"detector": bundle["detector"].to_dict(), "meta": meta}, indent=2)
    )
    print(
        f"Wrote {EXPORT_PATH.relative_to(config.ROOT)} ({EXPORT_PATH.stat().st_size / 1e3:.1f} KB)"
    )
    if report["detectors"]:
        export_web_benchmark(report)


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
