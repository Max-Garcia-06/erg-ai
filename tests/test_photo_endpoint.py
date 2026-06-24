import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from erg_ai.main import app
from erg_ai.middleware.auth import get_current_user_id
from erg_ai.db.session import init_db, reset_engine

EXTRACTED = {
    "meters": 5000,
    "elapsed_time": "20:15.3",
    "avg_split": "2:01.5",
    "avg_watts": 185,
    "stroke_rate": 22,
}


@pytest.fixture
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    reset_engine()
    init_db()
    yield TestClient(app)
    reset_engine()


def test_photo_endpoint_success(client):
    app.dependency_overrides[get_current_user_id] = lambda: "test-user"
    try:
        with patch("erg_ai.api.workouts.extract_erg_screen", return_value=EXTRACTED):
            response = client.post(
                "/api/workouts/photo",
                files={"image": ("screen.jpg", b"fake-jpeg", "image/jpeg")},
                data={"session_type": "steady_state"},
            )
        assert response.status_code == 200
        body = response.json()
        assert body["summary"]["avg_power"] == 185
        assert body["rating"]["rubric_id"] == "photo"
        assert body["chart_series"] == {"time": [], "watts": [], "pace": [], "stroke_rate": [], "heart_rate": []}
    finally:
        app.dependency_overrides.clear()


def test_photo_endpoint_bad_image(client):
    app.dependency_overrides[get_current_user_id] = lambda: "test-user"
    try:
        with patch("erg_ai.api.workouts.extract_erg_screen", side_effect=ValueError("No fields")):
            response = client.post(
                "/api/workouts/photo",
                files={"image": ("screen.jpg", b"blank", "image/jpeg")},
                data={"session_type": "steady_state"},
            )
        assert response.status_code == 422
    finally:
        app.dependency_overrides.clear()


def test_photo_endpoint_no_auth(client):
    # No override — real JWT dep fires, no token → 401
    response = client.post(
        "/api/workouts/photo",
        files={"image": ("screen.jpg", b"fake", "image/jpeg")},
        data={"session_type": "steady_state"},
    )
    assert response.status_code == 401
