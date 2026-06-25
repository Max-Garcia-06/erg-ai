"""SQLAlchemy ORM models."""

from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import DateTime, Float, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Workout(Base):
    __tablename__ = "workouts"
    __table_args__ = (Index("ix_workouts_user_uploaded", "user_id", "uploaded_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64), default="local", index=True)
    filename: Mapped[str] = mapped_column(String(512))
    uploaded_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    workout_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    session_type: Mapped[str] = mapped_column(String(32), index=True)
    detected_structure: Mapped[str] = mapped_column(String(32), default="unknown")

    row_count: Mapped[int] = mapped_column(Integer, default=0)
    duration_sec: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    summary_json: Mapped[str] = mapped_column(Text, default="{}")
    metrics_json: Mapped[str] = mapped_column(Text, default="{}")
    rating_json: Mapped[str] = mapped_column(Text, default="{}")
    coach_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    csv_path: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    source: Mapped[str] = mapped_column(String(16), default="csv", server_default="csv")

    title: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    def set_json_field(self, field: str, data: Dict[str, Any]) -> None:
        import json
        setattr(self, field, json.dumps(data))

    def get_json_field(self, field: str) -> Dict[str, Any]:
        import json
        raw = getattr(self, field) or "{}"
        return json.loads(raw) if raw else {}
