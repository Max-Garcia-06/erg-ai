"""Tests for workout comparison."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from erg_ai.db.session import init_db, reset_engine
from erg_ai.main import app

SAMPLE_CSV = Path(__file__).resolve().parent.parent / "sample_data" / "concept2-result-108710891.csv"


@pytest.fixture
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    reset_engine()
    init_db()
    return TestClient(app)


def _upload(client, session_type="steady_state"):
    with open(SAMPLE_CSV, "rb") as f:
        return client.post(
            "/api/workouts/analyze",
            data={"session_type": session_type},
            files={"file": ("w.csv", f, "text/csv")},
        )


def test_comparison_no_prior(client):
    if not SAMPLE_CSV.exists():
        pytest.skip("sample missing")
    wid = _upload(client).json()["workout_id"]
    detail = client.get(f"/api/workouts/{wid}").json()
    assert detail["comparison"]["has_prior"] is False


def test_comparison_with_prior(client):
    if not SAMPLE_CSV.exists():
        pytest.skip("sample missing")
    id1 = _upload(client).json()["workout_id"]
    id2 = _upload(client).json()["workout_id"]
    detail = client.get(f"/api/workouts/{id2}").json()
    cmp = detail["comparison"]
    assert cmp["has_prior"] is True
    assert cmp["previous"]["id"] == id1
    assert len(cmp["vs_previous"]) >= 1
    assert cmp["last_5_average"]["count"] >= 1
