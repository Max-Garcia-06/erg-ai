"""Workout coaching — template fallback and optional Gemini."""

import json
import logging
from typing import Any, Dict, List, Optional

from erg_ai.clients.gemini import generate_coach_feedback
from erg_ai.domain.workout_types import SESSION_TYPE_LABELS, SessionType

logger = logging.getLogger(__name__)


def _template_coach(
    rating: Dict[str, Any],
    summary: Dict[str, Any],
    session_type: SessionType,
) -> Dict[str, Any]:
    overall = rating.get("overall", 0)
    letter = rating.get("letter", "?")
    label = SESSION_TYPE_LABELS[session_type]
    focus = rating.get("focus_areas", [])
    warnings = rating.get("warnings", [])

    went_well = []
    if overall >= 80:
        went_well.append(f"Strong {label.lower()} session overall ({letter}, {overall}/100).")
    dims = rating.get("dimensions", {})
    for dim, sc in sorted(dims.items(), key=lambda x: -x[1]):
        if sc >= 80 and len(went_well) < 3:
            went_well.append(f"Good {dim.replace('_', ' ')} ({sc}/100).")

    if not went_well:
        went_well.append("You completed the piece — data captured for tracking.")

    work_on = [f["label"] for f in focus[:3]]
    if not work_on:
        work_on.append("Maintain current training load and consistency.")

    for w in warnings:
        work_on.append(w)

    next_session = (
        f"Next {label.lower()}: aim for steadier power in the opening third "
        f"if {work_on[0].lower()} was your weakest area."
        if focus
        else f"Repeat a similar {label.lower()} piece and compare scores over time."
    )

    return {
        "headline": f"{label}: {letter} ({overall}/100)",
        "went_well": went_well,
        "work_on": work_on,
        "next_session": next_session,
        "source": "template",
    }


def build_coach_feedback(
    session_type: SessionType,
    rating: Dict[str, Any],
    summary: Dict[str, Any],
    recent_summaries: Optional[List[Dict[str, Any]]] = None,
    use_gemini: bool = True,
) -> Dict[str, Any]:
    if use_gemini:
        gemini_result = generate_coach_feedback(
            session_type=session_type,
            rating=rating,
            summary=summary,
            recent_summaries=recent_summaries or [],
        )
        if gemini_result is not None:
            gemini_result["source"] = "gemini"
            return gemini_result

    result = _template_coach(rating, summary, session_type)
    return result


def coach_to_text(coach: Dict[str, Any]) -> str:
    return json.dumps(coach, indent=2)
