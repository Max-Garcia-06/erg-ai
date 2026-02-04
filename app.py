"""FastAPI backend for erg.ai rowing performance analytics."""

import logging
from pathlib import Path
from typing import Dict

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from infer_models import InferenceEngine
import yaml

# Setup logging
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

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize inference engine
try:
    engine = InferenceEngine()
    logger.info("Inference engine loaded successfully")
except Exception as e:
    logger.error(f"Failed to load inference engine: {e}")
    engine = None


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
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported"
        )

    try:
        # Read CSV
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        logger.info(f"Processing file: {file.filename} ({len(df)} rows)")

        # Normalize columns
        df.columns = df.columns.str.strip().str.lower()

        # Run inference
        if engine is None:
            logger.warning("Inference engine not available")
            raise HTTPException(
                status_code=503,
                detail="ML models not available"
            )

        results = engine.analyze(df)
        logger.info("Analysis completed successfully")

        return {
            "status": "success",
            "filename": file.filename,
            "results": results
        }

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