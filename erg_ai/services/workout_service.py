"""Orchestrate ingest, analyze, rate, and persist workouts."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sqlalchemy import desc
from sqlalchemy.orm import Session

from erg_ai.config import project_root
from erg_ai.db.models import Workout
from erg_ai.domain.workout_types import SESSION_TYPE_LABELS, SessionType
from erg_ai.services.analysis_service import analyze_dataframe
from erg_ai.services.coach_service import build_coach_feedback, coach_to_text
from erg_ai.services.csv_ingest import chart_series, derive_duration_sec, parse_csv_bytes
from erg_ai.services.rating_service import baseline_watts_for_type, score_workout


def _save_csv(workout_id: int, contents: bytes) -> str:
    uploads = project_root() / "data" / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    path = uploads / f"{workout_id}.csv"
    path.write_bytes(contents)
    return str(path)


def _load_csv(path: str) -> Optional[pd.DataFrame]:
    from erg_ai.services.csv_ingest import normalize_columns

    p = Path(path)
    if not p.exists():
        return None
    df = pd.read_csv(p)
    return normalize_columns(df)


def _recent_baselines(
    db: Session,
    user_id: str,
    session_type: str,
    exclude_id: Optional[int] = None,
    limit: int = 5,
) -> Tuple[List[float], List[Dict[str, Any]]]:
    q = (
        db.query(Workout)
        .filter(Workout.user_id == user_id, Workout.session_type == session_type)
        .order_by(desc(Workout.uploaded_at))
    )
    if exclude_id:
        q = q.filter(Workout.id != exclude_id)
    rows = q.limit(limit).all()
    powers = []
    summaries = []
    for w in rows:
        s = w.get_json_field("summary_json")
        if s.get("avg_power") is not None:
            powers.append(float(s["avg_power"]))
        summaries.append(s)
    return powers, summaries


def _global_median_power(db: Session, user_id: str, exclude_id: Optional[int] = None) -> Optional[float]:
    q = db.query(Workout).filter(Workout.user_id == user_id)
    if exclude_id:
        q = q.filter(Workout.id != exclude_id)
    powers = []
    for w in q.all():
        s = w.get_json_field("summary_json")
        if s.get("avg_power") is not None:
            powers.append(float(s["avg_power"]))
    if not powers:
        return None
    import numpy as np
    return float(np.median(powers))


def create_workout_from_csv(
    db: Session,
    contents: bytes,
    filename: str,
    session_type: SessionType,
    user_id: str = "local",
) -> Tuple[Workout, Dict[str, Any]]:
    df = parse_csv_bytes(contents, filename)
    metrics = analyze_dataframe(df)

    summary = {
        "avg_power": metrics["avg_power"],
        "avg_split": metrics["avg_split"],
        "consistency": metrics["consistency"],
        "drift": metrics["drift"],
        "interval_count": len(metrics.get("intervals", [])),
        "detected_structure": metrics["detected_structure"],
    }

    recent_powers, _ = _recent_baselines(db, user_id, session_type.value)
    global_med = _global_median_power(db, user_id)
    baseline = baseline_watts_for_type(recent_powers, global_med)
    rating = score_workout(session_type, metrics, baseline)

    workout = Workout(
        user_id=user_id,
        filename=filename,
        uploaded_at=datetime.now(UTC),
        workout_date=datetime.now(UTC),
        session_type=session_type.value,
        detected_structure=metrics["detected_structure"],
        row_count=metrics["row_count"],
        duration_sec=derive_duration_sec(df),
    )
    workout.set_json_field("summary_json", summary)
    workout.set_json_field("metrics_json", metrics)
    workout.set_json_field("rating_json", rating)

    db.add(workout)
    db.commit()
    db.refresh(workout)

    csv_path = _save_csv(workout.id, contents)
    workout.csv_path = csv_path
    db.commit()
    db.refresh(workout)

    chart = chart_series(df)
    return workout, chart


def rescore_workout(db: Session, workout: Workout, session_type: SessionType) -> Dict[str, Any]:
    metrics = workout.get_json_field("metrics_json")
    recent_powers, _ = _recent_baselines(
        db, workout.user_id, session_type.value, exclude_id=workout.id
    )
    global_med = _global_median_power(db, workout.user_id, exclude_id=workout.id)
    baseline = baseline_watts_for_type(recent_powers, global_med)
    rating = score_workout(session_type, metrics, baseline)
    workout.session_type = session_type.value
    workout.set_json_field("rating_json", rating)
    workout.coach_text = None
    db.commit()
    return rating


def get_workout_chart(db: Session, workout: Workout) -> Dict[str, Any]:
    if workout.csv_path:
        df = _load_csv(workout.csv_path)
        if df is not None:
            return chart_series(df)
    return {"time": [], "watts": [], "pace": [], "stroke_rate": [], "heart_rate": []}


def generate_coach_for_workout(db: Session, workout: Workout, force: bool = False) -> Dict[str, Any]:
    if workout.coach_text and not force:
        return json.loads(workout.coach_text)

    session_type = SessionType.from_str(workout.session_type)
    rating = workout.get_json_field("rating_json")
    summary = workout.get_json_field("summary_json")
    _, recent_summaries = _recent_baselines(
        db, workout.user_id, workout.session_type, exclude_id=workout.id, limit=3
    )

    coach = build_coach_feedback(
        session_type=session_type,
        rating=rating,
        summary=summary,
        recent_summaries=recent_summaries,
    )
    workout.coach_text = json.dumps(coach)
    db.commit()
    return coach
