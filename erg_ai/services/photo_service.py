"""Gemini Vision extraction and rating for photo-logged workouts."""

import json
import logging
import re
from typing import Any, Dict, Optional

from erg_ai.clients.gemini import get_gemini_api_key
from erg_ai.config import get_config

logger = logging.getLogger(__name__)

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover
    genai = None  # type: ignore[assignment]
    genai_types = None  # type: ignore[assignment]

_PROMPT = (
    "This is a Concept2 RowErg workout summary screen. "
    "Extract the following fields and return ONLY valid JSON with no markdown:\n"
    '{"meters": <integer or null>, "elapsed_time": <"MM:SS.T" string or null>, '
    '"avg_split": <"M:SS.T" string or null>, "avg_watts": <integer or null>, '
    '"stroke_rate": <integer or null>}\n'
    "Set any field to null if it is not visible or legible in the image."
)


def extract_erg_screen(image_bytes: bytes) -> Dict[str, Any]:
    """Call Gemini Vision to extract stats from an erg screen photo.

    Raises ValueError if all extracted fields are null (unreadable image).
    Raises RuntimeError if GEMINI_API_KEY is not set.
    """
    if genai is None or genai_types is None:
        raise RuntimeError("google-genai not installed")

    api_key = get_gemini_api_key()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")

    model_name = get_config().get("gemini_model", "gemini-2.5-flash")
    client = genai.Client(api_key=api_key)
    config = genai_types.GenerateContentConfig(
        response_mime_type="application/json",
        thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
    )

    response = client.models.generate_content(
        model=model_name,
        contents=[
            genai_types.Content(
                parts=[
                    genai_types.Part(
                        inline_data=genai_types.Blob(mime_type="image/jpeg", data=image_bytes)
                    ),
                    genai_types.Part(text=_PROMPT),
                ]
            )
        ],
        config=config,
    )

    text = _extract_response_text(response)
    if not text:
        raise ValueError("No text returned from vision model")
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()

    data: Dict[str, Any] = json.loads(text)
    data = _coerce_fields(data)
    if all(v is None for v in data.values()):
        raise ValueError("No fields extracted from image")
    return data


def _extract_response_text(response: Any) -> str:
    """Return model answer text, skipping thought parts when present."""
    if response.text:
        return response.text.strip()

    parts: list[str] = []
    candidates = getattr(response, "candidates", None) or []
    if candidates and candidates[0].content:
        for part in candidates[0].content.parts or []:
            if not part.text or getattr(part, "thought", False):
                continue
            parts.append(part.text)
    return "".join(parts).strip()


def _coerce_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce extracted fields to expected types, setting unparseable values to None."""
    def to_int(v: Any) -> Optional[int]:
        if v is None:
            return None
        try:
            return int(float(str(v).replace(",", "").split()[0]))
        except (ValueError, IndexError):
            return None

    return {
        "meters": to_int(data.get("meters")),
        "elapsed_time": str(data["elapsed_time"]) if data.get("elapsed_time") else None,
        "avg_split": str(data["avg_split"]) if data.get("avg_split") else None,
        "avg_watts": to_int(data.get("avg_watts")),
        "stroke_rate": to_int(data.get("stroke_rate")),
    }


def build_photo_summary(extracted: Dict[str, Any]) -> Dict[str, Any]:
    """Format extracted erg screen data for storage in summary_json."""
    return {
        "avg_power": extracted.get("avg_watts"),
        "avg_split": extracted.get("avg_split"),
        "meters": extracted.get("meters"),
        "elapsed_time": extracted.get("elapsed_time"),
        "stroke_rate": extracted.get("stroke_rate"),
        "consistency": None,
        "drift": None,
        "interval_count": 0,
        "detected_structure": "photo",
    }


def build_photo_rating(
    session_type_value: str,
    session_label: str,
    avg_watts: Optional[float],
) -> Dict[str, Any]:
    """Produce an effort-only rating dict compatible with score_workout output."""
    effort = min(100.0, max(0.0, float(avg_watts or 0) / 3.0))
    overall = round(effort, 1)
    if overall >= 90:
        letter = "A"
    elif overall >= 80:
        letter = "B"
    elif overall >= 70:
        letter = "C"
    elif overall >= 60:
        letter = "D"
    else:
        letter = "F"

    return {
        "overall": overall,
        "letter": letter,
        "session_type": session_type_value,
        "session_label": session_label,
        "rubric_id": "photo",
        "steady_state_format": None,
        "segment_count": 0,
        "dimensions": {"effort": round(effort, 1)},
        "weights": {"effort": 1.0},
        "focus_areas": [],
        "warnings": ["Photo log: effort score only — upload CSV for full analysis."],
        "scoring_notes": ["photo_log"],
    }
