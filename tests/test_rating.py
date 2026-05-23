"""Tests for type-aware rating."""

import pytest

from erg_ai.domain.workout_types import SessionType
from erg_ai.services.rating_service import score_workout


@pytest.fixture
def sample_metrics():
    return {
        "avg_power": 200,
        "consistency": 12,
        "drift": 5,
        "intervals": [[0, 100], [150, 250]],
        "detected_structure": "interval_pattern",
        "steady_state_format": "split_intervals",
        "interval_segments": [
            {"start": 0, "end": 100, "avg_watts": 198, "std_watts": 8, "drift_watts": 4, "row_count": 101},
            {"start": 150, "end": 250, "avg_watts": 202, "std_watts": 9, "drift_watts": 5, "row_count": 101},
        ],
        "segment_count": 2,
        "segment_avg_std_watts": 8.5,
        "segment_avg_drift_watts": 4.5,
        "segment_repeatability_cv": 0.01,
        "stroke_quality": {"overall": 78, "efficiency": 80, "consistency": 75, "drift": 70},
        "sq_pred": 76,
    }


def test_steady_state_split_intervals_not_penalized(sample_metrics):
    rating = score_workout(SessionType.STEADY_STATE, sample_metrics, baseline_watts=195)
    assert 0 <= rating["overall"] <= 100
    assert rating["session_type"] == "steady_state"
    assert rating["steady_state_format"] == "split_intervals"
    assert rating["segment_count"] == 2
    assert rating["dimensions"]["structure"] >= 70
    assert any("split steady state" in w.lower() for w in rating["warnings"])
    assert not any("did you mean" in w.lower() for w in rating["warnings"])


def test_steady_state_continuous_still_works():
    metrics = {
        "avg_power": 180,
        "consistency": 10,
        "drift": 3,
        "intervals": [],
        "detected_structure": "continuous",
        "steady_state_format": "continuous",
        "interval_segments": [],
        "segment_count": 0,
        "stroke_quality": {"overall": 80},
        "sq_pred": 82,
    }
    rating = score_workout(SessionType.STEADY_STATE, metrics)
    assert rating["dimensions"]["structure"] >= 80
    assert rating["warnings"] == []


def test_intervals_rewards_structure(sample_metrics):
    rating = score_workout(SessionType.INTERVALS, sample_metrics, baseline_watts=200)
    assert rating["dimensions"]["structure"] >= 50


def test_recovery_lenient_effort():
    metrics = {
        "avg_power": 120,
        "consistency": 8,
        "drift": 2,
        "intervals": [],
        "detected_structure": "continuous",
        "stroke_quality": {"overall": 70},
        "sq_pred": None,
    }
    rating = score_workout(SessionType.RECOVERY, metrics)
    assert rating["overall"] > 0
