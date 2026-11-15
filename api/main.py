"""HTTP API: upload a recording, get its anomalous readings back.

    uvicorn api.main:app --reload            # local, http://localhost:8000/docs

This file only does HTTP: parsing, limits, JSON. Detection lives in the
wearable_anomaly package so it stays testable without a server.

Environment variables:
    MODEL_PATH       served model JSON (default api/model/rolling_zscore.json)
    MAX_UPLOAD_MB    upload size limit (default 50)
    ALLOWED_ORIGINS  comma-separated origins allowed by CORS (default http://localhost:3000)
"""

import json
import os
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from wearable_anomaly.detect import detect_frame, parse_upload
from wearable_anomaly.detectors.rolling import RollingZScore
from wearable_anomaly.preprocess import clean

MODEL_PATH = Path(os.getenv("MODEL_PATH", Path(__file__).parent / "model" / "rolling_zscore.json"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "50")) * 1_000_000
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

_spec = json.loads(MODEL_PATH.read_text())
detector = RollingZScore.from_dict(_spec["detector"])
model_meta = _spec["meta"]

app = FastAPI(
    title="Wearable Sensor Anomaly Detection",
    description="Flags faulty readings in PAMAP2-format wearable sensor recordings.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "detector": detector.name,
        "sensors": len(detector.columns),
        "model": model_meta,
    }


@app.post("/detect")
def detect(
    file: Annotated[
        UploadFile, File(description="PAMAP2 .dat file, or CSV with the 31 sensor columns")
    ],
    max_flags: Annotated[
        int, Query(ge=0, le=20_000, description="Most anomalous readings to return")
    ] = 1000,
) -> dict:
    # A plain `def` endpoint runs in FastAPI's thread pool, so this CPU-heavy work
    # doesn't block the server from answering other requests.
    content = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File larger than {MAX_UPLOAD_BYTES // 1_000_000} MB")
    try:
        raw = parse_upload(content, file.filename or "upload.dat")
    except ValueError as e:
        raise HTTPException(400, str(e)) from e

    df = clean(raw)
    if df.empty:
        raise HTTPException(400, "No usable rows after cleaning (activity 0 rows are dropped)")

    flags = detect_frame(df, detector).drop(columns="predicted")
    top = (
        flags.nlargest(max_flags, "score")
        .sort_values(["timestamp", "sensor"])
        # sensor data is float32; widen before rounding so JSON shows 131.29, not 131.2899932861
        .astype({"timestamp": "float64", "value": "float64", "score": "float64"})
        .round({"timestamp": 2, "value": 4, "score": 4})
    )
    return {
        "filename": file.filename,
        "rows": len(df),
        "dropped_rows": len(raw) - len(df),
        "sensors": len(detector.columns),
        "flagged": len(flags),
        "flagged_pct": round(100 * len(flags) / (len(df) * len(detector.columns)), 3),
        "flagged_per_sensor": flags["sensor"].value_counts().to_dict(),
        "returned_flags": len(top),
        "flags": json.loads(top.round(4).to_json(orient="records")),
    }
