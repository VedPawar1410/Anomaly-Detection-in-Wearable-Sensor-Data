"""Benchmark detectors on unseen subjects with injected faults.

    python -m wearable_anomaly.benchmark                    # every registered detector
    python -m wearable_anomaly.benchmark --detectors per_sensor_rf --retrain

Fairness rules:
- Faults are injected once (fixed seed) and every detector scores the same data.
- Thresholds come only from the calibration subject; test subjects are never used to tune.
- Detectors without a saved model are trained first (models/<name>.joblib).

Writes reports/benchmark.json (everything) and reports/benchmark.md (summary table).
"""

import argparse
import json
import time

import numpy as np
from sklearn.metrics import average_precision_score

from wearable_anomaly import config, detectors, faults
from wearable_anomaly.detectors.per_sensor import Rule2024
from wearable_anomaly.train import load_or_train, load_splits, model_path


def detection_metrics(scores: np.ndarray, event_ids: np.ndarray, events) -> dict:
    """Metrics for one detector's scores (rows x sensors) against the injected truth."""
    truth = event_ids > 0
    flags = scores > 1
    tp = int((flags & truth).sum())
    precision = tp / max(int(flags.sum()), 1)
    recall = tp / int(truth.sum())
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    # Threshold-free: how well the scores rank faulty readings above clean ones.
    finite = np.nan_to_num(scores, posinf=1e9)
    pr_auc = average_precision_score(truth.ravel(), finite.ravel())

    hit_events = set(np.unique(event_ids[flags & truth]))
    per_type = {}
    for kind in faults.FAULT_TYPES:
        ev = events[events.type == kind]
        mask = np.isin(event_ids, ev.id.to_numpy())
        per_type[kind] = {
            "reading_recall": round(float(flags[mask].mean()), 4),
            "event_recall": round(float(ev.id.isin(hit_events).mean()), 4),
            "events": len(ev),
        }
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "pr_auc": round(float(pr_auc), 4),
        "flagged_pct": round(100 * float(flags.mean()), 3),
        "per_fault_type": per_type,
    }


def evaluate_one(name, detector, meta, test, corrupted, event_ids, events) -> dict:
    t0 = time.time()
    clean_scores = detector.score(test).to_numpy()
    score_s = time.time() - t0
    scores = detector.score(corrupted).to_numpy()
    result = {
        "clean_false_alarm_pct": round(100 * float((clean_scores > 1).mean()), 3),
        "detection": detection_metrics(scores, event_ids, events),
        "per_sensor_false_alarm_pct": {
            c: round(100 * float(v), 3)
            for c, v in zip(detector.columns, (clean_scores > 1).mean(axis=0), strict=True)
        },
        "score_us_per_row": round(1e6 * score_s / len(test), 2),
        "meta": meta,
    }
    path = model_path(name)
    if path.exists():
        result["model_mb"] = round(path.stat().st_size / 1e6, 1)
    if hasattr(detector, "extra_metrics"):
        result["extra"] = detector.extra_metrics(test)
    return result


def markdown_table(results: dict) -> str:
    kinds = faults.FAULT_TYPES
    head = "| Detector | F1 | PR-AUC | Precision | Recall | Clean false alarms | " + " | ".join(
        f"{k} recall" for k in kinds
    )
    lines = [head + " |", "|" + "---|" * (6 + len(kinds))]
    for name, r in results.items():
        d = r["detection"]
        per = " | ".join(f"{d['per_fault_type'][k]['event_recall']:.0%}" for k in kinds)
        lines.append(
            f"| {name} | {d['f1']:.3f} | {d['pr_auc']:.3f} | {d['precision']:.3f} | "
            f"{d['recall']:.3f} | {r['clean_false_alarm_pct']:.2f}% | {per} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detectors", nargs="+", default=sorted(detectors.REGISTRY))
    parser.add_argument("--retrain", action="store_true")
    args = parser.parse_args()

    splits = load_splits()
    test = splits["test"]
    cols = config.SENSOR_COLUMNS
    corrupted, event_ids, events = faults.inject(test, splits["train"][cols].std(), cols)
    print(
        f"Test subjects {config.TEST_SUBJECTS}: {len(test):,} rows, {len(events):,} injected "
        f"events covering {100 * (event_ids > 0).mean():.2f}% of readings"
    )

    bundles = {n: load_or_train(n, splits, args.retrain) for n in args.detectors}
    runs = {n: (b["detector"], b["meta"]) for n, b in bundles.items()}
    if "per_sensor_rf" in bundles:  # the 2024 decision rule on the same models
        runs[Rule2024.name] = (Rule2024(bundles["per_sensor_rf"]["detector"]), {})

    results = {}
    for name, (det, meta) in runs.items():
        print(f"Scoring {name}...")
        results[name] = evaluate_one(name, det, meta, test, corrupted, event_ids, events)

    report = {
        "test_subjects": config.TEST_SUBJECTS,
        "test_rows": len(test),
        "injected_events": len(events),
        "injected_reading_pct": round(100 * float((event_ids > 0).mean()), 3),
        "events_per_type": events.type.value_counts().to_dict(),
        "target_false_alarm_pct": 100 * config.TARGET_FALSE_ALARM_RATE,
        "detectors": results,
    }
    config.REPORTS_DIR.mkdir(exist_ok=True)
    (config.REPORTS_DIR / "benchmark.json").write_text(json.dumps(report, indent=2))
    table = markdown_table(results)
    (config.REPORTS_DIR / "benchmark.md").write_text(
        "# Benchmark\n\nGenerated by `python -m wearable_anomaly.benchmark`. Test subjects "
        f"{config.TEST_SUBJECTS}, {len(test):,} rows, {len(events):,} injected faults "
        f"({report['injected_reading_pct']}% of readings). Per-type columns are event recall "
        "(an event counts as caught if any of its readings is flagged).\n\n" + table + "\n"
    )
    print("\n" + table)
    print("\nWrote reports/benchmark.json and reports/benchmark.md")


if __name__ == "__main__":
    main()
