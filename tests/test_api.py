import io

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import main
from wearable_anomaly import config


@pytest.fixture
def client():
    return TestClient(main.app)


def _recording(n=3000, seed=0) -> pd.DataFrame:
    """Smooth synthetic signals for every sensor, like slow body movement."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 100
    data = {
        c: np.sin(2 * np.pi * (0.2 + 0.01 * i) * t) + rng.normal(0, 0.02, n)
        for i, c in enumerate(config.SENSOR_COLUMNS)
    }
    return pd.DataFrame({"timestamp": t, "activity_id": 4, **data})


def _csv(df) -> bytes:
    return df.to_csv(index=False).encode()


def test_health_reports_the_served_model(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["detector"] == "rolling_zscore"
    assert body["sensors"] == 31


def test_detect_csv_flags_an_injected_spike(client):
    df = _recording()
    df.loc[2000, "hand_acc16_x"] += 25.0
    r = client.post("/detect", files={"file": ("walk.csv", _csv(df), "text/csv")})
    assert r.status_code == 200
    body = r.json()
    assert body["rows"] == len(df)
    hit = [f for f in body["flags"] if f["sensor"] == "hand_acc16_x" and f["timestamp"] == 20.0]
    assert hit and hit[0]["score"] > 1


def test_detect_accepts_pamap2_dat_and_drops_transient_rows(client):
    df = _recording(1000)
    raw = pd.DataFrame(np.zeros((1000, len(config.RAW_COLUMNS))), columns=config.RAW_COLUMNS)
    for c in df.columns:
        raw[c] = df[c]
    raw.loc[:99, "activity_id"] = 0  # transient, dropped by clean()
    buf = io.StringIO()
    np.savetxt(buf, raw.to_numpy(), fmt="%.5f")
    r = client.post("/detect", files={"file": ("subject101.dat", buf.getvalue().encode())})
    assert r.status_code == 200
    assert r.json()["rows"] == 900 and r.json()["dropped_rows"] == 100


def test_max_flags_caps_the_response(client):
    df = _recording()
    df.loc[500:2500:10, "chest_gyro_y"] += 30.0  # many spikes
    r = client.post("/detect?max_flags=5", files={"file": ("x.csv", _csv(df))})
    body = r.json()
    assert body["returned_flags"] == 5 and body["flagged"] > 5


def test_rejects_csv_missing_sensor_columns(client):
    df = _recording().drop(columns=["heart_rate", "ankle_mag_z"])
    r = client.post("/detect", files={"file": ("bad.csv", _csv(df))})
    assert r.status_code == 400
    assert "missing 2 sensor columns" in r.json()["detail"]


def test_rejects_non_numeric_dat(client):
    r = client.post("/detect", files={"file": ("x.dat", b"hello world\nnot numbers\n")})
    assert r.status_code == 400


def test_rejects_files_over_the_size_limit(client, monkeypatch):
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 100)
    r = client.post("/detect", files={"file": ("big.csv", b"x" * 200)})
    assert r.status_code == 413
