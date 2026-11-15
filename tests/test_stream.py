import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from api import main
from wearable_anomaly import config
from wearable_anomaly.detectors.rolling import RollingZScore
from wearable_anomaly.stream import StreamingZScore, load_demo


def _signal(n, seed, cols=("a", "b", "c")):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 100
    df = pd.DataFrame(
        {
            c: np.sin(2 * np.pi * (0.3 + i / 10) * t) + rng.normal(0, 0.05, n)
            for i, c in enumerate(cols)
        },
        dtype=np.float32,
    )
    df.insert(0, "subject", 1)
    return df


def test_streaming_scores_match_batch_scores_exactly():
    """The served (streaming) detector must equal the benchmarked (batch) one: no serving skew."""
    det = RollingZScore(["a", "b", "c"]).fit(_signal(5000, 0)).calibrate(_signal(5000, 1))
    test = _signal(3000, 2)
    test.loc[1000, "a"] += 5.0  # spike
    test.loc[2000:2149, "b"] = test.loc[1999, "b"]  # stuck
    batch = det.score(test).to_numpy()
    online = StreamingZScore(det).update_batch(test[["a", "b", "c"]].to_numpy())
    np.testing.assert_allclose(online, batch, rtol=1e-5, atol=1e-6)


def test_streaming_matches_batch_on_the_real_demo_with_the_served_model():
    spec = json.loads((config.ROOT / "api" / "model" / "rolling_zscore.json").read_text())
    det = RollingZScore.from_dict(spec["detector"])
    demo, _ = load_demo()
    demo.insert(0, "subject", 108)
    first = demo.iloc[:3000]
    batch = det.score(first).to_numpy()
    online = StreamingZScore(det).update_batch(first[det.columns].to_numpy())
    np.testing.assert_allclose(online, batch, rtol=1e-5, atol=1e-5)


def test_websocket_sends_meta_then_batches():
    client = TestClient(main.app)
    with client.websocket_connect("/ws/stream?speed=50") as ws:
        meta = ws.receive_json()
        assert meta["type"] == "meta" and len(meta["sensors"]) == 31
        assert len(meta["faults"]) > 0
        batch = ws.receive_json()
        assert batch["type"] == "batch" and batch["seq"] == 0
        assert np.asarray(batch["values"]).shape == (10, 31)
        assert {"flags", "faults", "first_sample_time", "compute_ms"} <= set(batch)


def test_websocket_rejects_unknown_browser_origins():
    client = TestClient(main.app)
    with (
        pytest.raises(WebSocketDisconnect),
        client.websocket_connect("/ws/stream", headers={"origin": "https://evil.example"}) as ws,
    ):
        ws.receive_json()
