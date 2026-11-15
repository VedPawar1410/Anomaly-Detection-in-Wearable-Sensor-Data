"""Online (sample-by-sample) version of the rolling z-score, plus latency measurement.

    python -m wearable_anomaly.stream                      # compute latency over the demo
    python -m wearable_anomaly.stream --ws ws://localhost:8000/ws/stream --seconds 30
                                                           # end-to-end latency from a running server

The batch detector (detectors/rolling.py) uses pandas rolling windows over a whole
DataFrame. A live stream gets one sample at a time, so this class keeps a ring
buffer of the last `scale_window` samples and a stuck counter per sensor. It must
give the same scores as the batch version: tests/test_stream.py checks that, so
the benchmark numbers describe what is actually served (no training/serving skew).
"""

import argparse
import json
import time

import numpy as np
import pandas as pd

from wearable_anomaly import config
from wearable_anomaly.detectors.rolling import RollingZScore

BATCH_SIZE = 10  # samples per message: 10 messages/s at 100 Hz


class StreamingZScore:
    def __init__(self, detector: RollingZScore):
        self.cw, self.sw = detector.center_window, detector.scale_window
        self.floor = detector.scale_floor_
        self.tj, self.ts = detector.thresholds_
        n = len(detector.columns)
        self.buf = np.zeros((self.sw, n))  # ring buffer of past samples
        self.count = 0
        self.last = None
        self.run = np.zeros(n)

    def _history(self) -> np.ndarray:
        k = min(self.count, self.sw)
        if self.count <= self.sw:
            return self.buf[:k]
        p = self.count % self.sw
        return np.concatenate((self.buf[p:], self.buf[:p]))  # oldest -> newest

    def update(self, x) -> np.ndarray:
        """Score one sample (one value per sensor). Above 1 means anomaly."""
        x32 = np.asarray(x, dtype=np.float32)
        x = x32.astype(np.float64)
        past = self._history()
        recent = past[-self.cw :]
        center = np.median(recent, axis=0) if len(recent) >= 5 else np.full_like(x, np.nan)
        scale = past.std(axis=0, ddof=1) if len(past) >= 20 else np.full_like(x, np.nan)
        jump = np.nan_to_num(np.abs(x - center) / np.maximum(scale, self.floor), nan=0.0)

        self.run = np.where(x32 == self.last, self.run + 1, 1) if self.last is not None else 1.0
        self.last = x32
        self.buf[self.count % self.sw] = x
        self.count += 1
        return np.maximum(jump / self.tj, self.run / self.ts)

    def update_batch(self, rows: np.ndarray) -> np.ndarray:
        return np.vstack([self.update(r) for r in rows])


def load_demo():
    """The committed demo recording and its injected faults."""
    from wearable_anomaly.demo import DEMO_CSV, DEMO_FAULTS

    df = pd.read_csv(DEMO_CSV, dtype={c: np.float32 for c in config.SENSOR_COLUMNS})
    return df, json.loads(DEMO_FAULTS.read_text())


def _pct(a) -> dict:
    a = np.asarray(a)
    return {f"p{q}": round(float(np.percentile(a, q)), 3) for q in (50, 95, 99)}


def measure_compute(detector: RollingZScore) -> dict:
    df, _ = load_demo()
    values = df[detector.columns].to_numpy()
    online, times = StreamingZScore(detector), []
    t_all = time.perf_counter()
    for i in range(0, len(values), BATCH_SIZE):
        t0 = time.perf_counter()
        online.update_batch(values[i : i + BATCH_SIZE])
        times.append(1000 * (time.perf_counter() - t0))
    total = time.perf_counter() - t_all
    return {
        "batch_size": BATCH_SIZE,
        "batches": len(times),
        "compute_ms_per_batch": _pct(times),
        "throughput_samples_per_s": round(len(values) / total),
    }


def measure_websocket(url: str, seconds: float) -> dict:
    from websockets.sync.client import connect

    e2e, ready = [], []
    period = BATCH_SIZE / config.SAMPLE_RATE_HZ
    deadline = time.time() + seconds
    with connect(url, max_size=None) as ws:
        json.loads(ws.recv())  # meta
        while time.time() < deadline:
            msg = json.loads(ws.recv())
            if msg["type"] != "batch":
                break
            now = time.time()
            # oldest sample in the batch: includes waiting for the batch to fill (up to 90 ms)
            e2e.append(1000 * (now - msg["first_sample_time"]))
            # from the batch being complete to the client having its flags
            ready.append(1000 * (now - msg["first_sample_time"] - period * msg["speed_factor"]))
    return {
        "url": url,
        "messages": len(e2e),
        "oldest_sample_to_client_ms": _pct(e2e),
        "batch_ready_to_client_ms": _pct(ready),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ws", help="measure end to end against a running server's /ws/stream")
    parser.add_argument("--seconds", type=float, default=30)
    args = parser.parse_args()

    spec = json.loads((config.ROOT / "api" / "model" / "rolling_zscore.json").read_text())
    detector = RollingZScore.from_dict(spec["detector"])
    out = config.REPORTS_DIR / "stream_latency.json"
    report = json.loads(out.read_text()) if out.exists() else {}
    if args.ws:
        report["websocket"] = measure_websocket(args.ws, args.seconds)
    else:
        report["compute"] = measure_compute(detector)
    config.REPORTS_DIR.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
