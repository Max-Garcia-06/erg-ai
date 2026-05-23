"""Optional Google Gemini coaching client."""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from erg_ai.config import get_config
from erg_ai.domain.workout_types import SESSION_TYPE_LABELS, SessionType

logger = logging.getLogger(__name__)


def _get_api_key() -> Optional[str]:
    return os.environ.get("GEMINI_API_KEY") or get_config().get("gemini_api_key")


def generate_coach_feedback(
    session_type: SessionType,
    rating: Dict[str, Any],
    summary: Dict[str, Any],
    recent_summaries: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    api_key = _get_api_key()
    if not api_key:
        return None

    try:
        from google import genai
    except ImportError:
        logger.warning("google-genai not installed; using template coach")
        return None

    cfg = get_config()
    model_name = cfg.get("gemini_model", "gemini-2.0-flash")

    label = SESSION_TYPE_LABELS[session_type]
    prompt = f"""You are a rowing erg coach. The athlete did a "{label}" session.

Rating (deterministic, do not contradict):
{json.dumps(rating, indent=2)}

Workout summary:
{json.dumps(summary, indent=2)}

Recent same-type sessions (if any):
{json.dumps(recent_summaries, indent=2)}

Respond with ONLY valid JSON matching this schema:
{{
  "headline": "one short sentence",
  "went_well": ["bullet", "..."],
  "work_on": ["bullet", "..."],
  "next_session": "one concrete suggestion for the next session"
}}
Ground advice in the numbers. Be concise and actionable."""

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )
        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        data = json.loads(text)
        required = {"headline", "went_well", "work_on", "next_session"}
        if not required.issubset(data.keys()):
            logger.warning("Gemini response missing fields")
            return None
        return data
    except Exception as exc:
        logger.error("Gemini coach failed: %s", exc)
        return None
