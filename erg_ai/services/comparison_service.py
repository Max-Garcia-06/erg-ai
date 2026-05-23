"""Build workout comparisons vs prior same-type sessions."""

from typing import Any, Dict, List, Optional

import numpy as np
from sqlalchemy import desc
from sqlalchemy.orm import Session

from erg_ai.db.models import Workout
from erg_ai.domain.workout_types import SESSION_TYPE_LABELS, SessionType


def _snapshot(w: Workout) -> Dict[str, Any]:
    summary = w.get_json_field("summary_json")
    rating = w.get_json_field("rating_json")
    st = SessionType.from_str(w.session_type)
    return {
        "id": w.id,
        "filename": w.filename,
        "uploaded_at": w.uploaded_at.isoformat(),
        "session_type": w.session_type,
        "session_label": SESSION_TYPE_LABELS[st],
        "avg_power": summary.get("avg_power"),
        "overall_score": rating.get("overall"),
        "letter": rating.get("letter"),
        "consistency": summary.get("consistency"),
        "drift": summary.get("drift"),
        "avg_split": summary.get("avg_split"),
        "duration_sec": w.duration_sec,
    }


def _delta(
    metric_key: str,
    label: str,
    current: Optional[float],
    reference: Optional[float],
    higher_is_better: bool = True,
) -> Dict[str, Any]:
    if current is None or reference is None:
        return {
            "metric": metric_key,
            "label": label,
            "current": current,
            "reference": reference,
            "delta": None,
            "delta_pct": None,
            "direction": "na",
            "favorable": None,
        }

    delta = round(current - reference, 1)
    delta_pct = round((delta / reference) * 100, 1) if reference != 0 else None

    if abs(delta) < 0.05:
        direction = "same"
        favorable = True
    elif delta > 0:
        direction = "up"
        favorable = higher_is_better
    else:
        direction = "down"
        favorable = not higher_is_better

    return {
        "metric": metric_key,
        "label": label,
        "current": current,
        "reference": reference,
        "delta": delta,
        "delta_pct": delta_pct,
        "direction": direction,
        "favorable": favorable,
    }


def _aggregate(snapshots: List[Dict[str, Any]], field: str) -> Optional[float]:
    vals = [s[field] for s in snapshots if s.get(field) is not None]
    if not vals:
        return None
    return round(float(np.mean(vals)), 1)


def build_workout_comparison(
    db: Session,
    workout: Workout,
    user_id: str = "local",
    prior_limit: int = 5,
) -> Dict[str, Any]:
    """Compare workout to previous same-type session and prior-5 average."""
    priors = (
        db.query(Workout)
        .filter(
            Workout.user_id == user_id,
            Workout.session_type == workout.session_type,
            Workout.uploaded_at < workout.uploaded_at,
        )
        .order_by(desc(Workout.uploaded_at))
        .limit(prior_limit)
        .all()
    )

    current = _snapshot(workout)
    prior_snapshots = [_snapshot(p) for p in priors]
    previous = prior_snapshots[0] if prior_snapshots else None

    last_5_avg = {
        "count": len(prior_snapshots),
        "avg_power": _aggregate(prior_snapshots, "avg_power"),
        "overall_score": _aggregate(prior_snapshots, "overall_score"),
        "consistency": _aggregate(prior_snapshots, "consistency"),
        "drift": _aggregate(prior_snapshots, "drift"),
        "avg_split": _aggregate(prior_snapshots, "avg_split"),
    }

    vs_previous = []
    if previous:
        vs_previous = [
            _delta("overall_score", "Overall score", current.get("overall_score"), previous.get("overall_score"), True),
            _delta("avg_power", "Avg power", current.get("avg_power"), previous.get("avg_power"), True),
            _delta("consistency", "Consistency (σ)", current.get("consistency"), previous.get("consistency"), False),
            _delta("drift", "Power drift", current.get("drift"), previous.get("drift"), False),
        ]
        if current.get("avg_split") is not None and previous.get("avg_split") is not None:
            vs_previous.append(
                _delta("avg_split", "Avg split", current.get("avg_split"), previous.get("avg_split"), False)
            )

    vs_last_5_avg = []
    if prior_snapshots:
        vs_last_5_avg = [
            _delta(
                "overall_score",
                "Overall score",
                current.get("overall_score"),
                last_5_avg.get("overall_score"),
                True,
            ),
            _delta(
                "avg_power",
                "Avg power",
                current.get("avg_power"),
                last_5_avg.get("avg_power"),
                True,
            ),
            _delta(
                "consistency",
                "Consistency (σ)",
                current.get("consistency"),
                last_5_avg.get("consistency"),
                False,
            ),
            _delta(
                "drift",
                "Power drift",
                current.get("drift"),
                last_5_avg.get("drift"),
                False,
            ),
        ]

    return {
        "session_type": workout.session_type,
        "session_label": current["session_label"],
        "previous": previous,
        "prior_same_type": prior_snapshots,
        "last_5_average": last_5_avg,
        "vs_previous": vs_previous,
        "vs_last_5_average": vs_last_5_avg,
        "has_prior": bool(prior_snapshots),
    }
