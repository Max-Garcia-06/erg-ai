"""Database engine and session factory."""

from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine
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
        _engine = create_engine(
            _database_url(),
            connect_args={"check_same_thread": False},
        )
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
    return _engine


def get_session_factory():
    get_engine()
    return _SessionLocal


def init_db() -> None:
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    uploads = project_root() / "data" / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)


def get_db() -> Generator[Session, None, None]:
    factory = get_session_factory()
    db = factory()
    try:
        yield db
    finally:
        db.close()
