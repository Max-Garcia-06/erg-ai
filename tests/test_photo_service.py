import pytest
from unittest.mock import MagicMock, patch


GOOD_RESPONSE = '{"meters": 5000, "elapsed_time": "20:15.3", "avg_split": "2:01.5", "avg_watts": 185, "stroke_rate": 22}'
ALL_NULL_RESPONSE = '{"meters": null, "elapsed_time": null, "avg_split": null, "avg_watts": null, "stroke_rate": null}'


def _mock_gemini(text: str):
    mock_response = MagicMock()
    mock_response.text = text
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response
    return mock_client


def test_extract_erg_screen_success():
    from erg_ai.services.photo_service import extract_erg_screen
    mock_client = _mock_gemini(GOOD_RESPONSE)
    with patch("erg_ai.services.photo_service.genai.Client", return_value=mock_client):
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            result = extract_erg_screen(b"fake-image")
    assert result["meters"] == 5000
    assert result["avg_watts"] == 185
    assert result["stroke_rate"] == 22


def test_extract_erg_screen_all_null_raises():
    from erg_ai.services.photo_service import extract_erg_screen
    mock_client = _mock_gemini(ALL_NULL_RESPONSE)
    with patch("erg_ai.services.photo_service.genai.Client", return_value=mock_client):
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            with pytest.raises(ValueError, match="No fields"):
                extract_erg_screen(b"blank-image")


def test_build_photo_summary():
    from erg_ai.services.photo_service import build_photo_summary
    extracted = {"meters": 5000, "elapsed_time": "20:15.3", "avg_split": "2:01.5", "avg_watts": 185, "stroke_rate": 22}
    summary = build_photo_summary(extracted)
    assert summary["avg_power"] == 185
    assert summary["meters"] == 5000
    assert summary["detected_structure"] == "photo"


def test_build_photo_rating_letter():
    from erg_ai.services.photo_service import build_photo_rating
    rating = build_photo_rating("steady_state", "Steady State", 250)
    assert rating["letter"] in {"A", "B", "C", "D", "F"}
    assert rating["rubric_id"] == "photo"
    assert "photo_log" in rating["scoring_notes"]
