"""FastAPI backend for erg.ai rowing performance analytics."""

import io
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import yaml

from infer_models import InferenceEngine

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

logging.basicConfig(
    level=getattr(logging, config.get('log_level', 'INFO')),
    format=config.get('log_format'),
    handlers=[
        logging.FileHandler(config.get('log_file', 'erg_ai.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="erg.ai API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

try:
    engine = InferenceEngine()
    logger.info("Inference engine loaded successfully")
except Exception as e:
    logger.error(f"Failed to load inference engine: {e}")
    engine = None

# Alias map shared between training and inference pipelines
_COLUMN_ALIASES = {
    'time (s)': 'time', 'time (seconds)': 'time', 'time(s)': 'time',
    'pace (sec/500m)': 'pace', 'pace (seconds)': 'pace',
    'power': 'watts', 'power (watts)': 'watts',
    'stroke rate': 'stroke_rate', 'stroke-rate': 'stroke_rate', 'spm': 'stroke_rate',
    'heart rate': 'heart_rate', 'hr': 'heart_rate', 'bpm': 'heart_rate',
    'distance (meters)': 'distance', 'distance (m)': 'distance',
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and coerce numeric types."""
    df.columns = df.columns.str.strip().str.lower()
    df.rename(columns=_COLUMN_ALIASES, inplace=True)
    for col in ['watts', 'pace', 'stroke_rate', 'heart_rate', 'time', 'distance']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'watts' in df.columns:
        df = df.dropna(subset=['watts'])
    return df


def _compute_stroke_quality_components(df: pd.DataFrame) -> Dict:
    """Compute rule-based stroke quality sub-scores from config weights and ranges."""
    watts_arr = df["watts"].values

    rolling_std = float(pd.Series(watts_arr).rolling(10, min_periods=1).std().dropna().mean())
    watts_jitter = float(pd.Series(watts_arr).diff().abs().mean())

    sr_mean = (
        float(df["stroke_rate"].dropna().mean())
        if "stroke_rate" in df.columns and df["stroke_rate"].notna().any()
        else None
    )
    avg_watts = float(np.nanmean(watts_arr))
    efficiency = avg_watts / sr_mean if (sr_mean and sr_mean > 0) else avg_watts / 150.0

    eff_range = config.get('efficiency_range', [0.8, 2.5])
    cons_range = config.get('consistency_range', [5, 50])
    smooth_range = config.get('smoothness_range', [5, 50])

    norm_eff = float(np.clip(
        (efficiency - eff_range[0]) / (eff_range[1] - eff_range[0]), 0, 1
    ))
    norm_cons = float(np.clip(
        1 - (rolling_std - cons_range[0]) / (cons_range[1] - cons_range[0]), 0, 1
    ))
    norm_smooth = float(np.clip(
        1 - (watts_jitter - smooth_range[0]) / (smooth_range[1] - smooth_range[0]), 0, 1
    ))

    eff_w = config.get('efficiency_weight', 0.4)
    cons_w = config.get('consistency_weight', 0.35)
    smooth_w = config.get('smoothness_weight', 0.25)
    overall = (eff_w * norm_eff + cons_w * norm_cons + smooth_w * norm_smooth) * 100

    return {
        "overall": round(overall, 1),
        "efficiency": round(norm_eff * 100, 1),
        "consistency": round(norm_cons * 100, 1),
        # Key is "drift" to match the existing frontend field name; value is smoothness
        "drift": round(norm_smooth * 100, 1),
    }


@app.on_event("startup")
async def startup_event():
    logger.info("erg.ai server starting...")


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.get("/health")
async def health_check() -> Dict:
    return {
        "status": "healthy",
        "models_loaded": engine is not None
    }


@app.post("/analyze")
async def analyze_workout(file: UploadFile = File(...)) -> Dict:
    if not file.filename.endswith('.csv'):
        logger.warning(f"Invalid file type: {file.filename}")
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    contents = await file.read()

    max_bytes = config.get('max_file_size_mb', 10) * 1024 * 1024
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {config.get('max_file_size_mb', 10)}MB."
        )

    try:
        df = pd.read_csv(io.BytesIO(contents))
        logger.info(f"Processing file: {file.filename} ({len(df)} rows)")

        df = _normalize_columns(df)

        if 'watts' not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required 'watts'/'power' column. Found: {list(df.columns)}"
            )

        if engine is None:
            raise HTTPException(status_code=503, detail="ML models not available")

        ml_results = engine.analyze(df)

        # Summary statistics computed directly from the normalized DataFrame
        watts_vals = df["watts"].dropna()
        avg_power = float(watts_vals.mean())
        consistency = float(watts_vals.std())

        window = max(1, len(watts_vals) // 10)
        drift = float(watts_vals.iloc[-window:].mean() - watts_vals.iloc[:window].mean())

        avg_split = (
            float(df["pace"].dropna().mean())
            if "pace" in df.columns and df["pace"].notna().any()
            else None
        )

        sq_components = _compute_stroke_quality_components(df)

        # Unpack ML inference results into the flat structure the frontend expects
        interval_result = ml_results.get("intervals", {})
        interval_pairs: List = (
            [list(p) for p in interval_result.get("intervals", [])]
            if interval_result.get("available") else []
        )

        anomaly_result = ml_results.get("anomalies", {})
        anomaly_indices: List = (
            anomaly_result.get("anomalies", [])
            if anomaly_result.get("available") else []
        )

        sq_result = ml_results.get("stroke_quality", {})
        sq_pred: Optional[float] = sq_result.get("score") if sq_result.get("available") else None

        workout_type = "Interval" if len(interval_pairs) > 1 else "Steady State"

        logger.info("Analysis completed successfully")

        return {
            "status": "success",
            "filename": file.filename,
            "avg_power": round(avg_power, 1),
            "avg_split": round(avg_split, 1) if avg_split is not None else None,
            "consistency": round(consistency, 1),
            "drift": round(drift, 1),
            "workout_type": workout_type,
            "interval_report": (
                f"{len(interval_pairs)} interval(s) detected"
                if interval_pairs else "No intervals detected"
            ),
            "intervals": interval_pairs,
            "anomalies": anomaly_indices,
            "stroke_quality": sq_components,
            "sq_pred": round(sq_pred, 1) if sq_pred is not None else None,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    server_config = config.get('server', {})
    uvicorn.run(
        app,
        host=server_config.get('host', '0.0.0.0'),
        port=server_config.get('port', 8000),
        reload=server_config.get('reload', False)
    )
