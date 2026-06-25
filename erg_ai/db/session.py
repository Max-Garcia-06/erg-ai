"""Database engine and session factory."""

from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session, sessionmaker

from erg_ai.config import get_config, project_root
from erg_ai.db.models import Base

_engine = None
_SessionLocal = None


def _database_url() -> str:
    import os

    env_url = os.environ.get("DATABASE_URL")
    if env_url:
        return env_url
    cfg = get_config()
    url = cfg.get("database_url")
    if url:
        return url
    data_dir = project_root() / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{data_dir / 'erg.db'}"


def reset_engine() -> None:
    """Reset engine (for tests)."""
    global _engine, _SessionLocal
    _engine = None
    _SessionLocal = None


def get_engine():
    global _engine, _SessionLocal
    if _engine is None:
        url = _database_url()
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        _engine = create_engine(url, connect_args=connect_args)
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
    return _engine


def get_session_factory():
    get_engine()
    return _SessionLocal


def _migrate_schema(engine) -> None:
    """Apply lightweight schema updates create_all does not handle."""
    if "workouts" not in inspect(engine).get_table_names():
        return
    columns = {col["name"] for col in inspect(engine).get_columns("workouts")}
    if "source" not in columns:
        with engine.begin() as conn:
            conn.execute(
                text("ALTER TABLE workouts ADD COLUMN source VARCHAR(16) DEFAULT 'csv'")
            )
    if "title" not in columns:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE workouts ADD COLUMN title VARCHAR(200)"))
    if "notes" not in columns:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE workouts ADD COLUMN notes TEXT"))


def init_db() -> None:
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    _migrate_schema(engine)
    uploads = project_root() / "data" / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)


def get_db() -> Generator[Session, None, None]:
    factory = get_session_factory()
    db = factory()
    try:
        yield db
    finally:
        db.close()
