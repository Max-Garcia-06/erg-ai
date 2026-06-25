"""API integration tests."""

import io
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


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"


def test_session_types(client):
    res = client.get("/api/workouts/session-types")
    assert res.status_code == 200
    types = {t["value"] for t in res.json()}
    assert "steady_state" in types
    assert "threshold" in types


def test_analyze_and_list(client):
    if not SAMPLE_CSV.exists():
        pytest.skip("sample CSV missing")

    with open(SAMPLE_CSV, "rb") as f:
        res = client.post(
            "/api/workouts/analyze",
            data={"session_type": "steady_state"},
            files={"file": ("test.csv", f, "text/csv")},
        )
    assert res.status_code == 200, res.text
    data = res.json()
    assert "workout_id" in data
    assert data["rating"]["overall"] >= 0

    list_res = client.get("/api/workouts")
    assert list_res.status_code == 200
    assert any(w["id"] == data["workout_id"] for w in list_res.json())


def test_patch_rescore(client):
    if not SAMPLE_CSV.exists():
        pytest.skip("sample CSV missing")

    with open(SAMPLE_CSV, "rb") as f:
        created = client.post(
            "/api/workouts/analyze",
            data={"session_type": "steady_state"},
            files={"file": ("test.csv", f, "text/csv")},
        ).json()

    wid = created["workout_id"]
    patched = client.patch(
        f"/api/workouts/{wid}",
        json={"session_type": "intervals"},
    )
    assert patched.status_code == 200
    assert patched.json()["session_type"] == "intervals"


def test_default_title_and_notes_patch(client):
    if not SAMPLE_CSV.exists():
        pytest.skip("sample CSV missing")

    with open(SAMPLE_CSV, "rb") as f:
        created = client.post(
            "/api/workouts/analyze",
            data={"session_type": "steady_state"},
            files={"file": ("test.csv", f, "text/csv")},
        ).json()
    wid = created["workout_id"]

    detail = client.get(f"/api/workouts/{wid}").json()
    assert detail["title"].startswith("Steady State · ")
    assert detail["notes"] is None

    patched = client.patch(
        f"/api/workouts/{wid}",
        json={"title": "Morning grind", "notes": "Felt strong"},
    ).json()
    assert patched["title"] == "Morning grind"
    assert patched["notes"] == "Felt strong"
    assert patched["session_type"] == "steady_state"

    listed = next(w for w in client.get("/api/workouts").json() if w["id"] == wid)
    assert listed["title"] == "Morning grind"

    reset = client.patch(f"/api/workouts/{wid}", json={"title": ""}).json()
    assert reset["title"].startswith("Steady State · ")


def test_coach_endpoint(client):
    if not SAMPLE_CSV.exists():
        pytest.skip("sample CSV missing")

    with open(SAMPLE_CSV, "rb") as f:
        created = client.post(
            "/api/workouts/analyze",
            data={"session_type": "threshold"},
            files={"file": ("test.csv", f, "text/csv")},
        ).json()

    res = client.post(f"/api/workouts/{created['workout_id']}/coach")
    assert res.status_code == 200
    coach = res.json()["coach"]
    assert "headline" in coach
    assert "went_well" in coach
