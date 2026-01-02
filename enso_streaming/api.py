# enso_streaming/api.py
from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from .config import AppConfig
from .data_pipeline import DataPipeline
from .model import PredictionModel
from .utils import CsvLogger


# -----------------------------
# Models
# -----------------------------
class PredPoint(BaseModel):
    timestamp: str
    pred: float


class TailResponse(BaseModel):
    n: int
    preds: List[PredPoint]


STATE: Dict[str, Any] = {
    "lock": asyncio.Lock(),
    "task": None,
    "running": False,
    "cursor": 0,
    "buffer": [],  # list[dict(timestamp, pred)]
    "last_ts": None,
}


def _cfg_from_env() -> AppConfig:
    # Local-friendly defaults; override in Docker with /app/data paths
    return AppConfig(
        start=os.getenv("START", "2007-01-01"),
        end=os.getenv("END", "2017-12-31"),
        lead=int(os.getenv("LEAD", "1")),
        max_lag=int(os.getenv("MAX_LAG", "15")),
        sst_path=os.getenv("SST_PATH", "data/sst.mon.mean.trefadj.anom.1880to2018.nc"),
        enso_path=os.getenv("ENSO_PATH", "data/nino34.long.anom.data.txt"),
        model_path=os.getenv("MODEL_PATH", "data/linear_lag.joblib"),
        interval=float(os.getenv("INTERVAL", "10")),
        out_csv=os.getenv("OUT_CSV", None),  # optional
        show_features=False,
    )


async def _producer_loop() -> None:
    """
    Produce exactly ONE prediction every cfg.interval seconds,
    append to STATE["buffer"], optionally also append to CSV.
    """
    cfg: AppConfig = STATE["cfg"]
    feats_df: pd.DataFrame = STATE["feats_df"]
    model: PredictionModel = STATE["model"]
    logger: Optional[CsvLogger] = STATE.get("logger")

    while STATE["running"]:
        async with STATE["lock"]:
            i = int(STATE["cursor"])
            if i >= len(feats_df):
                # finished producing all rows; stop
                STATE["running"] = False
                break

            ts = feats_df.index[i]
            row = feats_df.iloc[i]
            pred = model.predict_row(row)

            item = {"timestamp": ts.strftime("%Y-%m-%d"), "pred": float(pred)}
            STATE["buffer"].append(item)
            STATE["last_ts"] = ts
            STATE["cursor"] = i + 1

            # optional CSV logging (same format as your old streamer)
            if logger is not None:
                logger.append(ts, float(pred))

        await asyncio.sleep(float(cfg.interval))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    cfg = _cfg_from_env()
    cfg.validate()

    pipeline = DataPipeline(cfg)
    feats_df, feature_names = pipeline.build_features()
    if feats_df.empty:
        raise RuntimeError("No features produced; check START/END, LEAD/MAX_LAG, and file coverage.")

    model = PredictionModel(cfg)

    logger: Optional[CsvLogger] = None
    if cfg.out_csv:
        out_path = Path(cfg.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        logger = CsvLogger(str(out_path))
        logger.init()

    STATE["cfg"] = cfg
    STATE["feats_df"] = feats_df
    STATE["feature_names"] = feature_names
    STATE["model"] = model
    STATE["logger"] = logger

    # reset stream state
    async with STATE["lock"]:
        STATE["buffer"] = []
        STATE["cursor"] = 0
        STATE["last_ts"] = None
        STATE["running"] = True

    # start producer task
    STATE["task"] = asyncio.create_task(_producer_loop())

    try:
        yield
    finally:
        # Shutdown
        async with STATE["lock"]:
            STATE["running"] = False
        task = STATE.get("task")
        if task is not None:
            task.cancel()
            try:
                await task
            except Exception:
                pass


app = FastAPI(title="ENSO Streaming API", version="2.0", lifespan=lifespan)


# -----------------------------
# Endpoints
# -----------------------------
@app.get("/health")
async def health() -> Dict[str, Any]:
    cfg: AppConfig = STATE["cfg"]
    feats_df: pd.DataFrame = STATE["feats_df"]

    async with STATE["lock"]:
        produced = len(STATE["buffer"])
        cursor = int(STATE["cursor"])
        running = bool(STATE["running"])
        last_ts = STATE["last_ts"]

    return {
        "status": "ok",
        "rows_total": int(len(feats_df)),
        "rows_produced": int(produced),
        "cursor": cursor,
        "running": running,
        "from": feats_df.index.min().strftime("%Y-%m-%d"),
        "to": feats_df.index.max().strftime("%Y-%m-%d"),
        "interval": float(cfg.interval),
        "lead": int(cfg.lead),
        "max_lag": int(cfg.max_lag),
        "model_path": str(cfg.model_path),
        "out_csv": str(cfg.out_csv) if cfg.out_csv else None,
        "last_timestamp": last_ts.strftime("%Y-%m-%d") if last_ts is not None else None,
    }


@app.get("/tail", response_model=TailResponse)
async def tail(n: int = Query(200, ge=1, le=5000)) -> TailResponse:
    async with STATE["lock"]:
        buf = STATE["buffer"][-int(n):]
    return TailResponse(n=len(buf), preds=[PredPoint(**x) for x in buf])


@app.get("/latest", response_model=PredPoint)
async def latest() -> PredPoint:
    async with STATE["lock"]:
        if not STATE["buffer"]:
            raise HTTPException(status_code=404, detail="No predictions produced yet.")
        item = STATE["buffer"][-1]
    return PredPoint(**item)


@app.post("/reset")
async def reset() -> Dict[str, Any]:
    """
    Restart streaming from the beginning (buffer cleared, cursor reset).
    """
    async with STATE["lock"]:
        STATE["buffer"] = []
        STATE["cursor"] = 0
        STATE["last_ts"] = None
        STATE["running"] = True

    # restart task
    task = STATE.get("task")
    if task is not None:
        task.cancel()
    STATE["task"] = asyncio.create_task(_producer_loop())

    return {"status": "ok", "message": "stream reset"}


# Invoke-RestMethod "http://localhost:8000/health"
# Invoke-RestMethod "http://localhost:8000/latest"
# Invoke-RestMethod "http://localhost:8000/tail?n=10"
