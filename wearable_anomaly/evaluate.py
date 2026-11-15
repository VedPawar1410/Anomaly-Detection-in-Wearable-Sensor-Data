"""Evaluate the trained detector on unseen test subjects.

    python -m wearable_anomaly.evaluate

Writes reports/metrics.json with:
- prediction quality per sensor (R², MAE) on clean test data
- false-alarm rate on clean test data (should be close to the calibration target)
- precision / recall / F1 on test data with injected faults, overall and per fault type
- the 2024 rule (|actual - pred| > 0.1 * actual) applied to the same model, for comparison
"""

import json

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

from wearable_anomaly import config, faults
from wearable_anomaly.train import MODEL_PATH, load_splits


def detection_metrics(flags: np.ndarray, event_ids: np.ndarray, events) -> dict:
    truth = event_ids > 0
    tp = int((flags & truth).sum())
    precision = tp / max(int(flags.sum()), 1)
    recall = tp / int(truth.sum())
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    out = {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "flagged_pct": round(100 * float(flags.mean()), 3),
    }
    # recall per fault type: by reading, and by event (an event counts if any reading is flagged)
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
    out["per_fault_type"] = per_type
    return out


def main() -> None:
    bundle = joblib.load(MODEL_PATH)
    detector = bundle["detector"]
    splits = load_splits()
    test, cols = splits["test"], detector.columns
    print(f"Evaluating on subjects {config.TEST_SUBJECTS}: {len(test):,} rows")

    # 1. prediction quality and false alarms on clean test data
    preds = detector.predict(test)
    clean_scores = detector.score(test, preds)
    per_sensor = {
        c: {
            "r2": round(float(r2_score(test[c], preds[c])), 4),
            "mae": round(float(mean_absolute_error(test[c], preds[c])), 4),
            "false_alarm_pct": round(100 * float((clean_scores[c] > 1).mean()), 3),
        }
        for c in cols
    }

    # 2. injected faults
    train_std = splits["train"][cols].std()
    corrupted, event_ids, events = faults.inject(test, train_std, cols)
    cpreds = detector.predict(corrupted)
    flags = (detector.score(corrupted, cpreds) > 1).to_numpy()
    actual, p = corrupted[cols].to_numpy(), cpreds.to_numpy()
    old_rule = np.abs(actual - p) > 0.1 * actual  # the 2024 rule, bug included

    metrics = {
        "detector": detector.name,
        "model_meta": bundle["meta"],
        "test_subjects": config.TEST_SUBJECTS,
        "test_rows": len(test),
        "injected_events": len(events),
        "injected_reading_pct": round(100 * float((event_ids > 0).mean()), 3),
        "clean_false_alarm_pct": round(100 * float((clean_scores > 1).to_numpy().mean()), 3),
        "target_false_alarm_pct": 100 * config.TARGET_FALSE_ALARM_RATE,
        "median_r2": round(float(np.median([v["r2"] for v in per_sensor.values()])), 4),
        "detection": detection_metrics(flags, event_ids, events),
        "detection_with_2024_rule": detection_metrics(old_rule, event_ids, events),
        "per_sensor": per_sensor,
    }
    config.REPORTS_DIR.mkdir(exist_ok=True)
    out = config.REPORTS_DIR / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2))

    d, o = metrics["detection"], metrics["detection_with_2024_rule"]
    print(
        f"Clean false alarms: {metrics['clean_false_alarm_pct']}% "
        f"(target {metrics['target_false_alarm_pct']}%)  |  median R² {metrics['median_r2']}"
    )
    print(f"Injected faults ({metrics['injected_reading_pct']}% of readings):")
    print(
        f"  fixed rule: P={d['precision']} R={d['recall']} F1={d['f1']} flagged={d['flagged_pct']}%"
    )
    print(
        f"  2024 rule:  P={o['precision']} R={o['recall']} F1={o['f1']} flagged={o['flagged_pct']}%"
    )
    for kind, v in d["per_fault_type"].items():
        print(
            f"  {kind:<7} reading recall {v['reading_recall']:.3f}  "
            f"event recall {v['event_recall']:.3f}  ({v['events']} events)"
        )
    print(f"Wrote {out.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
