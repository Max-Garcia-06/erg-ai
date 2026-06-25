"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager
from typing import Dict

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from erg_ai.api.workouts import router as workouts_router
from erg_ai.clients.gemini import get_gemini_api_key
from erg_ai.config import get_config, project_root
from erg_ai.db.session import init_db
from erg_ai.domain.workout_types import SessionType
from erg_ai.services.analysis_service import get_inference_engine
from erg_ai.services.workout_service import create_workout_from_csv

cfg = get_config()

logging.basicConfig(
    level=getattr(logging, cfg.get("log_level", "INFO")),
    format=cfg.get("log_format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
    handlers=[
        logging.FileHandler(cfg.get("log_file", "erg_ai.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

static_dir = project_root() / "static"


@asynccontextmanager
async def lifespan(application: FastAPI):
    init_db()
    get_inference_engine()
    logger.info("Training Partner server started")
    yield


app = FastAPI(title="Training Partner API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
app.include_router(workouts_router)


@app.get("/")
async def root():
    return FileResponse(str(static_dir / "index.html"))


@app.get("/health")
async def health_check() -> Dict:
    return {
        "status": "healthy",
        "models_loaded": get_inference_engine() is not None,
        "gemini_configured": bool(get_gemini_api_key()),
        "gemini_model": get_config().get("gemini_model", "gemini-2.5-flash"),
    }


@app.post("/analyze")
async def analyze_legacy(
    file: UploadFile = File(...),
    session_type: str = "steady_state",
):
    """Legacy analyze endpoint — delegates to workout pipeline with default session type."""
    from erg_ai.db.session import get_session_factory

    contents = await file.read()
    filename = file.filename or "workout.csv"
    try:
        st = SessionType.from_str(session_type)
    except ValueError:
        st = SessionType.STEADY_STATE

    factory = get_session_factory()
    db = factory()
    try:
        workout, _ = create_workout_from_csv(db, contents, filename, st)
        metrics = workout.get_json_field("metrics_json")
        rating = workout.get_json_field("rating_json")
        return {
            "status": "success",
            "workout_id": workout.id,
            "filename": workout.filename,
            **metrics,
            "rating": rating,
        }
    finally:
        db.close()


if __name__ == "__main__":
    import uvicorn

    server = cfg.get("server", {})
    uvicorn.run(
        "erg_ai.main:app",
        host=server.get("host", "0.0.0.0"),
        port=server.get("port", 8000),
        reload=server.get("reload", False),
    )
