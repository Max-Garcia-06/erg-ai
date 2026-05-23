"""Type-aware workout scoring."""

from typing import Any, Dict, List, Optional

import numpy as np

from erg_ai.config import get_config
from erg_ai.domain.workout_types import SESSION_TYPE_LABELS, SessionType


def _clip_score(value: float) -> float:
    return float(np.clip(value, 0, 100))


def _letter_band(score: float) -> str:
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


def _is_steady_state_splits(metrics: Dict[str, Any], rubric: Dict) -> bool:
    struct_cfg = rubric.get("structure", {})
    if struct_cfg.get("mode") != "steady_state_splits":
        return False
    segments = metrics.get("interval_segments") or []
    if len(segments) >= 2:
        return True
    return metrics.get("steady_state_format") == "split_intervals"


def _score_from_watts_std(watts_std: float) -> float:
    cons_range = get_config().get("consistency_range", [5, 50])
    low, high = cons_range[0], cons_range[1]
    raw = 1 - (watts_std - low) / max(high - low, 1)
    return _clip_score(raw * 100)


def _score_technique(metrics: Dict[str, Any]) -> float:
    sq = metrics.get("stroke_quality", {})
    overall = sq.get("overall")
    sq_pred = metrics.get("sq_pred")
    if overall is not None and sq_pred is not None:
        return _clip_score(0.5 * overall + 0.5 * sq_pred)
    if overall is not None:
        return _clip_score(overall)
    if sq_pred is not None:
        return _clip_score(sq_pred)
    return 50.0


def _score_consistency(metrics: Dict[str, Any], rubric: Dict) -> float:
    cons_cfg = rubric.get("consistency", {})
    segments = metrics.get("interval_segments") or []

    if cons_cfg.get("use_segment_std_when_split", False) and len(segments) >= 1:
        watts_std = metrics.get("segment_avg_std_watts")
        if watts_std is not None:
            return _score_from_watts_std(float(watts_std))

    return _score_from_watts_std(float(metrics.get("consistency", 20)))


def _score_pacing(metrics: Dict[str, Any], rubric: Dict) -> float:
    pacing_cfg = rubric.get("pacing", {})
    max_drift = pacing_cfg.get("max_drift_watts", 20)
    segments = metrics.get("interval_segments") or []

    if pacing_cfg.get("use_segment_drift_when_split", False) and len(segments) >= 1:
        drift = metrics.get("segment_avg_drift_watts")
        if drift is None:
            drift = float(
                np.mean([abs(s["drift_watts"]) for s in segments])
            )
        drift = abs(float(drift))
    else:
        drift = abs(metrics.get("drift", 0))

    if max_drift <= 0:
        return 100.0
    raw = 1 - drift / max_drift
    return _clip_score(raw * 100)


def _score_steady_state_structure(metrics: Dict[str, Any], rubric: Dict) -> float:
    """Score structure for SS — continuous pieces or split into multiple SS blocks."""
    struct_cfg = rubric.get("structure", {})
    segments: List[Dict[str, Any]] = metrics.get("interval_segments") or []
    n = len(segments)

    if n == 0:
        pairs = metrics.get("intervals", [])
        if len(pairs) <= 1:
            return 88.0
        return _clip_score(75.0)

    if n == 1:
        seg = segments[0]
        within = _score_from_watts_std(seg["std_watts"])
        max_drift = struct_cfg.get("max_within_segment_drift_watts", 12)
        drift = abs(seg["drift_watts"])
        pacing = _clip_score((1 - drift / max(max_drift, 1)) * 100)
        return _clip_score(0.55 * within + 0.45 * pacing)

    within_scores = [_score_from_watts_std(s["std_watts"]) for s in segments]
    within_score = float(np.mean(within_scores))

    avgs = [s["avg_watts"] for s in segments]
    mean_w = float(np.mean(avgs))
    cv = float(np.std(avgs) / mean_w) if mean_w > 0 else 1.0
    cv_max = struct_cfg.get("work_watts_cv_max", 0.08)
    repeat_score = _clip_score((1 - cv / max(cv_max, 0.01)) * 100)

    max_drift = struct_cfg.get("max_within_segment_drift_watts", 12)
    drift_scores = [
        _clip_score((1 - abs(s["drift_watts"]) / max(max_drift, 1)) * 100)
        for s in segments
    ]
    segment_pacing = float(np.mean(drift_scores))

    min_segments = struct_cfg.get("min_segments", 2)
    count_bonus = _clip_score(min(100, 70 + 8 * min(n, 6))) if n >= min_segments else 70.0

    return _clip_score(
        0.35 * within_score + 0.30 * repeat_score + 0.25 * segment_pacing + 0.10 * count_bonus
    )


def _score_structure(metrics: Dict[str, Any], rubric: Dict) -> float:
    struct_cfg = rubric.get("structure", {})

    if struct_cfg.get("mode") == "steady_state_splits":
        return _score_steady_state_structure(metrics, rubric)

    expect_intervals = struct_cfg.get("expect_intervals", False)
    min_pairs = struct_cfg.get("min_interval_pairs", 2)
    pairs = metrics.get("intervals", [])
    n_pairs = len(pairs)

    if expect_intervals:
        if n_pairs < min_pairs:
            return _clip_score(30 + 15 * n_pairs)
        segments = metrics.get("interval_segments") or []
        if len(segments) >= 2:
            avgs = [s["avg_watts"] for s in segments]
            mean_w = float(np.mean(avgs))
            cv = float(np.std(avgs) / mean_w) if mean_w > 0 else 1.0
            cv_max = struct_cfg.get("work_watts_cv_max", 0.15)
            repeat_score = _clip_score((1 - cv / max(cv_max, 0.01)) * 100)
        else:
            repeat_score = 70.0
        count_score = _clip_score(min(100, 50 + 10 * n_pairs))
        return _clip_score(0.5 * count_score + 0.5 * repeat_score)

    if n_pairs > 1:
        return _clip_score(max(40, 80 - 15 * n_pairs))
    return 85.0


def _score_effort(
    metrics: Dict[str, Any],
    rubric: Dict,
    baseline_watts: Optional[float],
) -> float:
    effort_cfg = rubric.get("effort", {})
    mode = effort_cfg.get("score_mode", "standard")
    avg_power = metrics.get("avg_power", 0)

    if mode == "lenient_low_power":
        return _clip_score(60 + min(40, avg_power / 5))

    if baseline_watts is None or baseline_watts <= 0:
        return _clip_score(min(100, 50 + avg_power / 4))

    ratio = avg_power / baseline_watts
    target = effort_cfg.get("target_ratio", 1.0)
    tolerance = effort_cfg.get("tolerance", 0.15)
    deviation = abs(ratio - target)
    raw = 1 - deviation / tolerance
    return _clip_score(raw * 100)


def _structure_warnings(session_type: SessionType, metrics: Dict[str, Any]) -> List[str]:
    pairs = len(metrics.get("intervals", []))
    segments = metrics.get("interval_segments") or []
    warnings: List[str] = []

    if session_type == SessionType.STEADY_STATE:
        if len(segments) >= 2:
            warnings.append(
                f"Scored as split steady state ({len(segments)} work segments) — "
                "consistency and pacing measured within each piece."
            )
        return warnings

    if session_type == SessionType.INTERVALS and pairs < 2:
        warnings.append(
            "Few intervals detected — check session type or ensure work/rest structure is clear."
        )
    return warnings


def score_workout(
    session_type: SessionType,
    metrics: Dict[str, Any],
    baseline_watts: Optional[float] = None,
) -> Dict[str, Any]:
    cfg = get_config()
    rubrics = cfg.get("session_types", {})
    rubric = rubrics.get(session_type.value, {})
    weights = rubric.get("weights", {})

    dimension_scores = {
        "technique": _score_technique(metrics),
        "consistency": _score_consistency(metrics, rubric),
        "pacing": _score_pacing(metrics, rubric),
        "structure": _score_structure(metrics, rubric),
        "effort": _score_effort(metrics, rubric, baseline_watts),
    }

    active_weights = {k: v for k, v in weights.items() if k in dimension_scores and v > 0}
    total_w = sum(active_weights.values())
    if total_w <= 0:
        active_weights = {"technique": 0.4, "consistency": 0.3, "pacing": 0.3}
        total_w = 1.0

    overall = sum(
        dimension_scores[k] * (active_weights[k] / total_w)
        for k in active_weights
    )
    overall = round(_clip_score(overall), 1)

    sorted_dims = sorted(
        [(k, dimension_scores[k]) for k in active_weights],
        key=lambda x: x[1],
    )
    focus_areas = []
    for dim, sc in sorted_dims[:4]:
        if sc < 75:
            focus_areas.append(
                {
                    "id": dim if dim != "structure" else "interval_structure",
                    "label": dim.replace("_", " ").title(),
                    "score": round(sc, 1),
                }
            )

    warnings = _structure_warnings(session_type, metrics)
    scoring_notes: List[str] = []
    if session_type == SessionType.STEADY_STATE and _is_steady_state_splits(metrics, rubric):
        n = len(metrics.get("interval_segments") or [])
        scoring_notes.append(f"split_steady_state_{n}_segments")

    return {
        "overall": overall,
        "letter": _letter_band(overall),
        "session_type": session_type.value,
        "session_label": SESSION_TYPE_LABELS[session_type],
        "rubric_id": session_type.value,
        "steady_state_format": metrics.get("steady_state_format"),
        "segment_count": metrics.get("segment_count", 0),
        "dimensions": {k: round(v, 1) for k, v in dimension_scores.items()},
        "weights": active_weights,
        "focus_areas": focus_areas[:4],
        "warnings": warnings,
        "scoring_notes": scoring_notes,
    }


def baseline_watts_for_type(
    recent_avg_powers: List[float],
    global_median: Optional[float] = None,
) -> Optional[float]:
    if recent_avg_powers:
        return float(np.median(recent_avg_powers))
    return global_median
