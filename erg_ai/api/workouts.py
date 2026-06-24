"""Workout REST API."""

import json
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from erg_ai.db.models import Workout
from erg_ai.db.session import get_db
from erg_ai.domain.workout_types import SESSION_TYPE_LABELS, SessionType
from erg_ai.schemas.workout import (
    CoachResponse,
    SessionTypeInfo,
    WorkoutAnalyzeResponse,
    WorkoutComparison,
    WorkoutDetailResponse,
    WorkoutListItem,
    WorkoutPatchRequest,
)
from erg_ai.services.comparison_service import build_workout_comparison
from erg_ai.services.workout_service import (
    create_workout_from_csv,
    generate_coach_for_workout,
    get_workout_chart,
    rescore_workout,
)

router = APIRouter(prefix="/api/workouts", tags=["workouts"])


@router.get("/session-types", response_model=List[SessionTypeInfo])
def list_session_types() -> List[SessionTypeInfo]:
    return [
        SessionTypeInfo(value=st.value, label=SESSION_TYPE_LABELS[st])
        for st in SessionType
    ]


@router.post("/analyze", response_model=WorkoutAnalyzeResponse)
async def analyze_workout(
    file: UploadFile = File(...),
    session_type: str = Form(...),
    db: Session = Depends(get_db),
) -> WorkoutAnalyzeResponse:
    try:
        st = SessionType.from_str(session_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    contents = await file.read()
    filename = file.filename or "workout.csv"

    workout, chart = create_workout_from_csv(db, contents, filename, st)
    summary = workout.get_json_field("summary_json")
    metrics = workout.get_json_field("metrics_json")
    rating = workout.get_json_field("rating_json")

    return WorkoutAnalyzeResponse(
        workout_id=workout.id,
        filename=workout.filename,
        session_type=workout.session_type,
        session_label=SESSION_TYPE_LABELS[st],
        summary=summary,
        metrics=metrics,
        rating=rating,
        chart_series=chart,
    )


@router.get("", response_model=List[WorkoutListItem])
def list_workouts(
    session_type: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    user_id: str = "local",
    db: Session = Depends(get_db),
) -> List[WorkoutListItem]:
    q = db.query(Workout).filter(Workout.user_id == user_id)
    if session_type:
        q = q.filter(Workout.session_type == session_type)
    rows = q.order_by(Workout.uploaded_at.desc()).offset(offset).limit(limit).all()

    items = []
    for w in rows:
        summary = w.get_json_field("summary_json")
        rating = w.get_json_field("rating_json")
        st = SessionType.from_str(w.session_type)
        items.append(
            WorkoutListItem(
                id=w.id,
                filename=w.filename,
                uploaded_at=w.uploaded_at,
                session_type=w.session_type,
                session_label=SESSION_TYPE_LABELS[st],
                source=w.source,
                avg_power=summary.get("avg_power"),
                overall_score=rating.get("overall"),
                letter=rating.get("letter"),
                focus_areas=rating.get("focus_areas", []),
            )
        )
    return items


@router.get("/{workout_id}", response_model=WorkoutDetailResponse)
def get_workout(
    workout_id: int,
    user_id: str = "local",
    db: Session = Depends(get_db),
) -> WorkoutDetailResponse:
    w = db.query(Workout).filter(Workout.id == workout_id, Workout.user_id == user_id).first()
    if not w:
        raise HTTPException(status_code=404, detail="Workout not found")

    st = SessionType.from_str(w.session_type)
    coach = None
    if w.coach_text:
        coach = json.loads(w.coach_text)

    comparison_data = build_workout_comparison(db, w, user_id=user_id)

    return WorkoutDetailResponse(
        id=w.id,
        filename=w.filename,
        uploaded_at=w.uploaded_at,
        session_type=w.session_type,
        session_label=SESSION_TYPE_LABELS[st],
        source=w.source,
        detected_structure=w.detected_structure,
        duration_sec=w.duration_sec,
        summary=w.get_json_field("summary_json"),
        metrics=w.get_json_field("metrics_json"),
        rating=w.get_json_field("rating_json"),
        chart_series=get_workout_chart(db, w),
        coach=coach,
        comparison=WorkoutComparison(**comparison_data),
    )


@router.get("/{workout_id}/compare", response_model=WorkoutComparison)
def get_workout_compare(
    workout_id: int,
    user_id: str = "local",
    db: Session = Depends(get_db),
) -> WorkoutComparison:
    w = db.query(Workout).filter(Workout.id == workout_id, Workout.user_id == user_id).first()
    if not w:
        raise HTTPException(status_code=404, detail="Workout not found")
    return WorkoutComparison(**build_workout_comparison(db, w, user_id=user_id))


@router.patch("/{workout_id}", response_model=WorkoutDetailResponse)
def patch_workout(
    workout_id: int,
    body: WorkoutPatchRequest,
    user_id: str = "local",
    db: Session = Depends(get_db),
) -> WorkoutDetailResponse:
    w = db.query(Workout).filter(Workout.id == workout_id, Workout.user_id == user_id).first()
    if not w:
        raise HTTPException(status_code=404, detail="Workout not found")

    try:
        st = SessionType.from_str(body.session_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    rescore_workout(db, w, st)
    return get_workout(workout_id, user_id, db)


@router.post("/{workout_id}/coach", response_model=CoachResponse)
def post_coach(
    workout_id: int,
    force: bool = False,
    user_id: str = "local",
    db: Session = Depends(get_db),
) -> CoachResponse:
    w = db.query(Workout).filter(Workout.id == workout_id, Workout.user_id == user_id).first()
    if not w:
        raise HTTPException(status_code=404, detail="Workout not found")

    coach = generate_coach_for_workout(db, w, force=force)
    return CoachResponse(workout_id=w.id, coach=coach)


@router.delete("/{workout_id}")
def delete_workout(
    workout_id: int,
    user_id: str = "local",
    db: Session = Depends(get_db),
) -> dict:
    w = db.query(Workout).filter(Workout.id == workout_id, Workout.user_id == user_id).first()
    if not w:
        raise HTTPException(status_code=404, detail="Workout not found")

    if w.csv_path:
        from pathlib import Path
        p = Path(w.csv_path)
        if p.exists():
            p.unlink()

    db.delete(w)
    db.commit()
    return {"status": "deleted", "id": workout_id}
