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
        self.d = detector
        self.cw, self.sw = detector.center_window, detector.scale_window
        self.floor = detector.scale_floor_
        self.thresholds = detector.thresholds_
        # the buffer must hold the longest look-back of any signal
        self.size = self.sw
        if detector.uses_level:
            self.size = max(
                self.sw,
                detector.level_scale_window,
                detector.level_lag + detector.level_window - 1,
            )
        n = len(detector.columns)
        self.buf = np.zeros((self.size, n))  # ring buffer of past samples
        self.count = 0
        self.last = None
        self.run = np.zeros(n)

    def _history(self) -> np.ndarray:
        k = min(self.count, self.size)
        if self.count <= self.size:
            return self.buf[:k]
        p = self.count % self.size
        return np.concatenate((self.buf[p:], self.buf[:p]))  # oldest -> newest

    def update(self, x) -> np.ndarray:
        """Score one sample (one value per sensor). Above 1 means anomaly."""
        nan = np.full(len(self.floor), np.nan)
        x32 = np.asarray(x, dtype=np.float32)
        x = x32.astype(np.float64)
        past = self._history()

        # jump: against the median of the previous cw samples, scaled by the previous sw
        recent = past[-self.cw :]
        center = np.median(recent, axis=0) if len(recent) >= 5 else nan
        jump_past = past[-self.sw :]
        scale = jump_past.std(axis=0, ddof=1) if len(jump_past) >= 20 else nan
        signals = [np.nan_to_num(np.abs(x - center) / np.maximum(scale, self.floor), nan=0.0)]

        # stuck: identical values in a row, including this one
        self.run = np.where(x32 == self.last, self.run + 1, 1) if self.last is not None else 1.0
        signals.append(np.where(self.d.stuck_mask, 0.0, np.broadcast_to(self.run, x.shape)))

        # level: last cw samples (incl. this one) vs a baseline that ended level_lag ago
        if self.d.uses_level:
            lag, bw, lsw = self.d.level_lag, self.d.level_window, self.d.level_scale_window
            recent_incl = np.vstack([past[-(self.cw - 1) :], x[None, :]])
            now_med = np.median(recent_incl, axis=0) if len(recent_incl) >= 5 else nan
            L = len(past)  # past[L - j] is the sample j steps back
            hi = L - lag + 1  # baseline covers lag ... lag + bw - 1 steps back
            base = past[max(0, hi - bw) : max(0, hi)]
            base_med = np.median(base, axis=0) if len(base) >= bw // 4 else nan
            long_past = past[-lsw:]
            long_scale = long_past.std(axis=0, ddof=1) if len(long_past) >= 200 else nan
            level = np.abs(now_med - base_med) / np.maximum(long_scale, self.floor)
            signals.append(np.nan_to_num(level, nan=0.0))

        self.last = x32
        self.buf[self.count % self.size] = x
        self.count += 1
        return np.max([s / t for s, t in zip(signals, self.thresholds, strict=True)], axis=0)

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

    model_file = config.ROOT / "api" / "model" / f"{config.SERVED_DETECTOR}.json"
    spec = json.loads(model_file.read_text())
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
