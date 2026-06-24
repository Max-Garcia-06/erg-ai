# Mobile Photo Log Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Supabase auth + Postgres, a Gemini Vision photo endpoint, and an Expo React Native mobile app that lets athletes log Concept2 RowErg workouts by photographing the summary screen.

**Architecture:** The existing FastAPI backend gains a JWT middleware (Supabase), a `source` column on workouts, and a new `POST /api/workouts/photo` endpoint backed by Gemini Vision. A new Expo app (separate repo) handles auth, camera capture, history, and CSV upload.

**Tech Stack:** FastAPI, SQLAlchemy, Supabase (Postgres + Auth), google-genai, python-jose, Expo SDK 52, expo-router, expo-camera, @supabase/supabase-js, React Query, Zustand

## Global Constraints

- Python 3.11 (pinned for Render)
- `google-genai` SDK already in requirements — use `from google import genai`
- Supabase JWTs: HS256, `aud="authenticated"`, `sub` = user UUID string
- `DATABASE_URL` env var already checked in `session.py` — Supabase Postgres URL goes there
- Mobile app is a **new repo** (`erg-ai-mobile`), not a subdirectory of this repo
- Photo workouts use `source="photo"`; CSV workouts use `source="csv"` (default)
- Rating for photo workouts is effort-only (no stroke data available)
- Existing endpoints keep working without JWT (backward-compat with the web UI)

---

## File Map

### Backend (this repo)

| Action | Path | Responsibility |
|--------|------|---------------|
| Modify | `erg_ai/db/models.py` | Add `source` column to Workout |
| Modify | `erg_ai/db/session.py` | Fix `connect_args` for Postgres; normalize `postgres://` URL |
| Modify | `erg_ai/schemas/workout.py` | Add `source` field to list + detail response schemas |
| Create | `erg_ai/middleware/__init__.py` | Package init |
| Create | `erg_ai/middleware/auth.py` | Supabase JWT → user_id dependency (strict + optional) |
| Create | `erg_ai/services/photo_service.py` | Gemini Vision call, field extraction, photo rating |
| Modify | `erg_ai/api/workouts.py` | Add `/photo` endpoint; swap `user_id="local"` → JWT optional dep |
| Modify | `requirements.txt` | Add `python-jose[cryptography]`, `psycopg2-binary` |

### Mobile (new repo: `erg-ai-mobile`)

| Action | Path | Responsibility |
|--------|------|---------------|
| Create | `lib/supabase.ts` | Supabase client singleton |
| Create | `lib/store.ts` | Zustand auth session store |
| Create | `lib/api.ts` | Typed fetch wrapper (injects JWT) |
| Create | `app/_layout.tsx` | Root layout: auth guard, session listener |
| Create | `app/(auth)/_layout.tsx` | Auth stack layout |
| Create | `app/(auth)/login.tsx` | Email + password login |
| Create | `app/(auth)/signup.tsx` | Email + password sign up |
| Create | `app/(tabs)/_layout.tsx` | Bottom tab bar (Log, History) |
| Create | `app/(tabs)/index.tsx` | Camera screen (Log tab) |
| Create | `app/preview.tsx` | Photo preview + session type picker + submit |
| Create | `app/(tabs)/history.tsx` | Workout history list |
| Create | `app/workout/[id].tsx` | Workout detail (photo tier vs CSV tier) |
| Create | `app/upload.tsx` | CSV upload modal |
| Create | `components/WorkoutCard.tsx` | History list item |

---

## Phase 1: Backend

### Task 1: Postgres compatibility + source column

**Files:**
- Modify: `erg_ai/db/models.py`
- Modify: `erg_ai/db/session.py`
- Modify: `erg_ai/schemas/workout.py`
- Test: `tests/test_db.py`

**Interfaces:**
- Produces: `Workout.source: str` column (values `"photo"` | `"csv"`, default `"csv"`)
- Produces: `WorkoutListItem.source: str`, `WorkoutDetailResponse.source: str`

- [ ] **Step 1: Write failing test for Postgres URL normalization**

```python
# tests/test_db.py
import pytest
from unittest.mock import patch
from erg_ai.db.session import reset_engine, get_engine


def test_sqlite_connect_args():
    reset_engine()
    with patch.dict("os.environ", {"DATABASE_URL": "sqlite:///test.db"}):
        engine = get_engine()
        assert engine.url.drivername == "sqlite"
    reset_engine()


def test_postgres_url_normalized():
    reset_engine()
    with patch.dict("os.environ", {"DATABASE_URL": "postgres://user:pass@host/db"}):
        engine = get_engine()
        assert str(engine.url).startswith("postgresql")
    reset_engine()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_db.py -v
```

Expected: FAIL — `postgres://` URL causes engine error or `connect_args` error

- [ ] **Step 3: Fix session.py for Postgres**

Replace the `get_engine` function in `erg_ai/db/session.py`:

```python
def get_engine():
    global _engine, _SessionLocal
    if _engine is None:
        url = _database_url()
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        _engine = create_engine(url, connect_args=connect_args)
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
    return _engine
```

- [ ] **Step 4: Add `source` column to Workout model**

In `erg_ai/db/models.py`, add after the `csv_path` field:

```python
source: Mapped[str] = mapped_column(String(16), default="csv", server_default="csv")
```

- [ ] **Step 5: Add `source` to response schemas**

In `erg_ai/schemas/workout.py`, add `source: str = "csv"` to `WorkoutListItem` and `WorkoutDetailResponse`:

```python
class WorkoutListItem(BaseModel):
    id: int
    filename: str
    uploaded_at: datetime
    session_type: str
    session_label: str
    source: str = "csv"          # ← add this line
    avg_power: Optional[float] = None
    overall_score: Optional[float] = None
    letter: Optional[str] = None
    focus_areas: List[Dict[str, Any]] = Field(default_factory=list)


class WorkoutDetailResponse(BaseModel):
    id: int
    filename: str
    uploaded_at: datetime
    session_type: str
    session_label: str
    source: str = "csv"          # ← add this line
    detected_structure: str
    duration_sec: Optional[float] = None
    summary: Dict[str, Any]
    metrics: Dict[str, Any]
    rating: Dict[str, Any]
    chart_series: Dict[str, Any]
    coach: Optional[Dict[str, Any]] = None
    comparison: Optional[WorkoutComparison] = None
```

- [ ] **Step 6: Expose `source` in the list and detail endpoints**

In `erg_ai/api/workouts.py`, `list_workouts`: add `source=w.source` to `WorkoutListItem(...)` constructor.

In `get_workout`: add `source=w.source` to `WorkoutDetailResponse(...)` constructor.

- [ ] **Step 7: Run tests**

```bash
pytest tests/test_db.py -v
```

Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add erg_ai/db/models.py erg_ai/db/session.py erg_ai/schemas/workout.py erg_ai/api/workouts.py tests/test_db.py
git commit -m "feat: add source column, Postgres URL normalization"
```

---

### Task 2: JWT auth dependency

**Files:**
- Create: `erg_ai/middleware/__init__.py`
- Create: `erg_ai/middleware/auth.py`
- Modify: `requirements.txt`
- Test: `tests/test_auth.py`

**Interfaces:**
- Produces: `get_current_user_id(credentials) -> str` — raises 401 if no/invalid JWT
- Produces: `get_optional_user_id(credentials) -> str` — returns `"local"` if no JWT

- [ ] **Step 1: Write failing tests**

```python
# tests/test_auth.py
import pytest
from unittest.mock import patch
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from jose import jwt

SECRET = "test-secret-at-least-32-chars-long!!"


def _make_token(secret: str = SECRET, sub: str = "user-123", aud: str = "authenticated") -> str:
    return jwt.encode({"sub": sub, "aud": aud}, secret, algorithm="HS256")


def test_get_current_user_id_valid():
    from erg_ai.middleware.auth import get_current_user_id
    token = _make_token()
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
    with patch("erg_ai.middleware.auth._get_jwt_secret", return_value=SECRET):
        user_id = get_current_user_id(creds)
    assert user_id == "user-123"


def test_get_current_user_id_no_token():
    from erg_ai.middleware.auth import get_current_user_id
    with pytest.raises(HTTPException) as exc:
        get_current_user_id(None)
    assert exc.value.status_code == 401


def test_get_current_user_id_bad_token():
    from erg_ai.middleware.auth import get_current_user_id
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="not.a.token")
    with patch("erg_ai.middleware.auth._get_jwt_secret", return_value=SECRET):
        with pytest.raises(HTTPException) as exc:
            get_current_user_id(creds)
    assert exc.value.status_code == 401


def test_get_optional_user_id_no_token():
    from erg_ai.middleware.auth import get_optional_user_id
    result = get_optional_user_id(None)
    assert result == "local"


def test_get_optional_user_id_valid():
    from erg_ai.middleware.auth import get_optional_user_id
    token = _make_token()
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
    with patch("erg_ai.middleware.auth._get_jwt_secret", return_value=SECRET):
        result = get_optional_user_id(creds)
    assert result == "user-123"
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_auth.py -v
```

Expected: FAIL — module not found

- [ ] **Step 3: Add python-jose to requirements.txt**

```
python-jose[cryptography]==3.3.0
psycopg2-binary==2.9.10
```

- [ ] **Step 4: Install new dependencies**

```bash
pip install "python-jose[cryptography]==3.3.0" "psycopg2-binary==2.9.10"
```

- [ ] **Step 5: Create `erg_ai/middleware/__init__.py`**

```python
```

(empty file)

- [ ] **Step 6: Create `erg_ai/middleware/auth.py`**

```python
"""Supabase JWT → FastAPI user_id dependency."""

import os
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

_security = HTTPBearer(auto_error=False)


def _get_jwt_secret() -> str:
    secret = os.environ.get("SUPABASE_JWT_SECRET", "")
    if not secret:
        raise RuntimeError("SUPABASE_JWT_SECRET env var not set")
    return secret


def get_current_user_id(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_security),
) -> str:
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    try:
        payload = jwt.decode(
            credentials.credentials,
            _get_jwt_secret(),
            algorithms=["HS256"],
            audience="authenticated",
        )
        user_id: Optional[str] = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        return user_id
    except JWTError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token") from exc


def get_optional_user_id(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_security),
) -> str:
    """Returns user_id from JWT if present, 'local' otherwise (backward-compat)."""
    if credentials is None:
        return "local"
    try:
        payload = jwt.decode(
            credentials.credentials,
            _get_jwt_secret(),
            algorithms=["HS256"],
            audience="authenticated",
        )
        return payload.get("sub") or "local"
    except (JWTError, RuntimeError):
        return "local"
```

- [ ] **Step 7: Run tests**

```bash
pytest tests/test_auth.py -v
```

Expected: all 5 PASS

- [ ] **Step 8: Commit**

```bash
git add erg_ai/middleware/ requirements.txt tests/test_auth.py
git commit -m "feat: add Supabase JWT auth dependency"
```

---

### Task 3: Photo service

**Files:**
- Create: `erg_ai/services/photo_service.py`
- Test: `tests/test_photo_service.py`

**Interfaces:**
- Produces: `extract_erg_screen(image_bytes: bytes) -> Dict[str, Any]` — calls Gemini Vision, returns `{meters, elapsed_time, avg_split, avg_watts, stroke_rate}` (values may be None); raises `ValueError` if all null
- Produces: `build_photo_summary(extracted: Dict) -> Dict` — formats for `summary_json`
- Produces: `build_photo_rating(session_type_value: str, session_label: str, avg_watts: Optional[float]) -> Dict` — effort-only rating dict

- [ ] **Step 1: Write failing tests**

```python
# tests/test_photo_service.py
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
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_photo_service.py -v
```

Expected: FAIL — module not found

- [ ] **Step 3: Create `erg_ai/services/photo_service.py`**

```python
"""Gemini Vision extraction and rating for photo-logged workouts."""

import base64
import json
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

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
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            types.Content(
                parts=[
                    types.Part(
                        inline_data=types.Blob(mime_type="image/jpeg", data=image_b64)
                    ),
                    types.Part(text=_PROMPT),
                ]
            )
        ],
    )

    text = response.text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])

    data: Dict[str, Any] = json.loads(text)
    if all(v is None for v in data.values()):
        raise ValueError("No fields extracted from image")
    return data


def build_photo_summary(extracted: Dict[str, Any]) -> Dict[str, Any]:
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
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_photo_service.py -v
```

Expected: all 4 PASS

- [ ] **Step 5: Commit**

```bash
git add erg_ai/services/photo_service.py tests/test_photo_service.py
git commit -m "feat: add photo_service for Gemini Vision erg screen extraction"
```

---

### Task 4: Photo endpoint + wire up optional JWT

**Files:**
- Modify: `erg_ai/api/workouts.py`
- Test: `tests/test_photo_endpoint.py`

**Interfaces:**
- Consumes: `get_optional_user_id` from `erg_ai.middleware.auth`
- Consumes: `get_current_user_id` from `erg_ai.middleware.auth`
- Consumes: `extract_erg_screen`, `build_photo_summary`, `build_photo_rating` from `erg_ai.services.photo_service`
- Produces: `POST /api/workouts/photo` → `WorkoutAnalyzeResponse`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_photo_endpoint.py
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from erg_ai.main import app
from erg_ai.middleware.auth import get_current_user_id

client = TestClient(app)

EXTRACTED = {
    "meters": 5000,
    "elapsed_time": "20:15.3",
    "avg_split": "2:01.5",
    "avg_watts": 185,
    "stroke_rate": 22,
}


def test_photo_endpoint_success():
    # Override JWT dep so we don't need a real token in tests
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


def test_photo_endpoint_bad_image():
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


def test_photo_endpoint_no_auth():
    # No override — real JWT dep fires, no token → 401
    response = client.post(
        "/api/workouts/photo",
        files={"image": ("screen.jpg", b"fake", "image/jpeg")},
        data={"session_type": "steady_state"},
    )
    assert response.status_code == 401
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_photo_endpoint.py -v
```

Expected: FAIL — endpoint not found

- [ ] **Step 3: Add photo endpoint and optional JWT to workouts.py**

At the top of `erg_ai/api/workouts.py`, add imports:

```python
from datetime import UTC, datetime

from erg_ai.middleware.auth import get_current_user_id, get_optional_user_id
from erg_ai.services.photo_service import (
    build_photo_rating,
    build_photo_summary,
    extract_erg_screen,
)
```

Replace ALL occurrences of `user_id: str = "local"` in existing endpoint signatures with `user_id: str = Depends(get_optional_user_id)`. There are five endpoints to update: `list_workouts`, `get_workout`, `get_workout_compare`, `patch_workout`, `post_coach`, `delete_workout`.

Then add the new endpoint **before** the `list_workouts` route (after `analyze_workout`):

```python
@router.post("/photo", response_model=WorkoutAnalyzeResponse)
async def analyze_photo_workout(
    image: UploadFile = File(...),
    session_type: str = Form(...),
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
) -> WorkoutAnalyzeResponse:
    try:
        st = SessionType.from_str(session_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    image_bytes = await image.read()

    try:
        extracted = extract_erg_screen(image_bytes)
    except (ValueError, Exception) as exc:
        raise HTTPException(
            status_code=422,
            detail="Could not read erg screen — try better lighting or a straighter angle",
        ) from exc

    summary = build_photo_summary(extracted)
    rating = build_photo_rating(st.value, SESSION_TYPE_LABELS[st], extracted.get("avg_watts"))

    workout = Workout(
        user_id=user_id,
        filename="photo_log",
        uploaded_at=datetime.now(UTC),
        workout_date=datetime.now(UTC),
        session_type=st.value,
        detected_structure="photo",
        source="photo",
        row_count=0,
    )
    workout.set_json_field("summary_json", summary)
    workout.set_json_field("metrics_json", {})
    workout.set_json_field("rating_json", rating)

    db.add(workout)
    db.commit()
    db.refresh(workout)

    return WorkoutAnalyzeResponse(
        workout_id=workout.id,
        filename=workout.filename,
        session_type=workout.session_type,
        session_label=SESSION_TYPE_LABELS[st],
        summary=summary,
        metrics={},
        rating=rating,
        chart_series={"time": [], "watts": [], "pace": [], "stroke_rate": [], "heart_rate": []},
    )
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_photo_endpoint.py tests/test_auth.py tests/test_db.py -v
```

Expected: all PASS

- [ ] **Step 5: Run full test suite to check for regressions**

```bash
pytest -v
```

Expected: all previously passing tests still PASS

- [ ] **Step 6: Commit**

```bash
git add erg_ai/api/workouts.py tests/test_photo_endpoint.py
git commit -m "feat: add POST /api/workouts/photo endpoint with Gemini Vision OCR"
```

---

## Phase 2: Mobile App

> All mobile tasks are executed in a **new directory** `erg-ai-mobile` created at the same level as `erg-ai`. This is a separate repo.

### Task 5: Expo scaffold + auth screens

**Files:**
- Create: `package.json`, `app.json`, `tsconfig.json` (via `create-expo-app`)
- Create: `lib/supabase.ts`
- Create: `lib/store.ts`
- Create: `app/_layout.tsx`
- Create: `app/(auth)/_layout.tsx`
- Create: `app/(auth)/login.tsx`
- Create: `app/(auth)/signup.tsx`

**Interfaces:**
- Produces: `useAuthStore()` hook — `{ session, setSession, clearSession }`
- Produces: `supabase` — typed Supabase client with AsyncStorage persistence
- Produces: Auth guard in root layout — redirects unauthenticated users to `/login`

- [ ] **Step 1: Scaffold Expo project**

```bash
npx create-expo-app@latest erg-ai-mobile -t tabs --no-install
cd erg-ai-mobile
```

- [ ] **Step 2: Install dependencies**

```bash
npm install
npx expo install expo-camera expo-document-picker
npm install @supabase/supabase-js zustand @tanstack/react-query
npx expo install @react-native-async-storage/async-storage
```

- [ ] **Step 3: Create environment config**

Create `.env.local`:
```
EXPO_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
EXPO_PUBLIC_SUPABASE_ANON_KEY=your-anon-key
EXPO_PUBLIC_API_URL=https://your-render-app.onrender.com
```

- [ ] **Step 4: Create `lib/supabase.ts`**

```typescript
import AsyncStorage from "@react-native-async-storage/async-storage";
import { createClient } from "@supabase/supabase-js";

const supabaseUrl = process.env.EXPO_PUBLIC_SUPABASE_URL!;
const supabaseAnonKey = process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY!;

export const supabase = createClient(supabaseUrl, supabaseAnonKey, {
  auth: {
    storage: AsyncStorage,
    autoRefreshToken: true,
    persistSession: true,
    detectSessionInUrl: false,
  },
});
```

- [ ] **Step 5: Create `lib/store.ts`**

```typescript
import { Session } from "@supabase/supabase-js";
import { create } from "zustand";

interface AuthState {
  session: Session | null;
  setSession: (session: Session | null) => void;
  clearSession: () => void;
}

export const useAuthStore = create<AuthState>((set) => ({
  session: null,
  setSession: (session) => set({ session }),
  clearSession: () => set({ session: null }),
}));
```

- [ ] **Step 6: Create root layout `app/_layout.tsx`**

```typescript
import { useEffect } from "react";
import { Stack, useRouter, useSegments } from "expo-router";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { supabase } from "../lib/supabase";
import { useAuthStore } from "../lib/store";

const queryClient = new QueryClient();

function AuthGuard({ children }: { children: React.ReactNode }) {
  const { session, setSession } = useAuthStore();
  const segments = useSegments();
  const router = useRouter();

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => setSession(session));
    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
      setSession(session);
    });
    return () => subscription.unsubscribe();
  }, []);

  useEffect(() => {
    const inAuthGroup = segments[0] === "(auth)";
    if (!session && !inAuthGroup) {
      router.replace("/(auth)/login");
    } else if (session && inAuthGroup) {
      router.replace("/(tabs)");
    }
  }, [session, segments]);

  return <>{children}</>;
}

export default function RootLayout() {
  return (
    <QueryClientProvider client={queryClient}>
      <AuthGuard>
        <Stack screenOptions={{ headerShown: false }} />
      </AuthGuard>
    </QueryClientProvider>
  );
}
```

- [ ] **Step 7: Create `app/(auth)/_layout.tsx`**

```typescript
import { Stack } from "expo-router";

export default function AuthLayout() {
  return <Stack screenOptions={{ headerShown: false }} />;
}
```

- [ ] **Step 8: Create `app/(auth)/login.tsx`**

```typescript
import { useState } from "react";
import { View, Text, TextInput, TouchableOpacity, StyleSheet, Alert } from "react-native";
import { Link } from "expo-router";
import { supabase } from "../../lib/supabase";

export default function LoginScreen() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);

  async function signIn() {
    setLoading(true);
    const { error } = await supabase.auth.signInWithPassword({ email, password });
    if (error) Alert.alert("Login failed", error.message);
    setLoading(false);
  }

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Training Partner</Text>
      <TextInput
        style={styles.input}
        placeholder="Email"
        autoCapitalize="none"
        keyboardType="email-address"
        value={email}
        onChangeText={setEmail}
      />
      <TextInput
        style={styles.input}
        placeholder="Password"
        secureTextEntry
        value={password}
        onChangeText={setPassword}
      />
      <TouchableOpacity style={styles.btn} onPress={signIn} disabled={loading}>
        <Text style={styles.btnText}>{loading ? "Signing in…" : "Sign in"}</Text>
      </TouchableOpacity>
      <Link href="/(auth)/signup" style={styles.link}>
        No account? Sign up
      </Link>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, justifyContent: "center", padding: 24, gap: 12 },
  title: { fontSize: 28, fontWeight: "700", textAlign: "center", marginBottom: 24 },
  input: { borderWidth: 1, borderColor: "#ccc", borderRadius: 8, padding: 12, fontSize: 16 },
  btn: { backgroundColor: "#0066cc", borderRadius: 8, padding: 14, alignItems: "center" },
  btnText: { color: "#fff", fontSize: 16, fontWeight: "600" },
  link: { textAlign: "center", color: "#0066cc", marginTop: 8 },
});
```

- [ ] **Step 9: Create `app/(auth)/signup.tsx`**

```typescript
import { useState } from "react";
import { View, Text, TextInput, TouchableOpacity, StyleSheet, Alert } from "react-native";
import { Link } from "expo-router";
import { supabase } from "../../lib/supabase";

export default function SignupScreen() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);

  async function signUp() {
    setLoading(true);
    const { error } = await supabase.auth.signUp({ email, password });
    if (error) {
      Alert.alert("Sign up failed", error.message);
    } else {
      Alert.alert("Check your email", "Click the confirmation link to activate your account.");
    }
    setLoading(false);
  }

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Create account</Text>
      <TextInput
        style={styles.input}
        placeholder="Email"
        autoCapitalize="none"
        keyboardType="email-address"
        value={email}
        onChangeText={setEmail}
      />
      <TextInput
        style={styles.input}
        placeholder="Password (min 6 chars)"
        secureTextEntry
        value={password}
        onChangeText={setPassword}
      />
      <TouchableOpacity style={styles.btn} onPress={signUp} disabled={loading}>
        <Text style={styles.btnText}>{loading ? "Creating…" : "Create account"}</Text>
      </TouchableOpacity>
      <Link href="/(auth)/login" style={styles.link}>
        Already have an account? Sign in
      </Link>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, justifyContent: "center", padding: 24, gap: 12 },
  title: { fontSize: 28, fontWeight: "700", textAlign: "center", marginBottom: 24 },
  input: { borderWidth: 1, borderColor: "#ccc", borderRadius: 8, padding: 12, fontSize: 16 },
  btn: { backgroundColor: "#0066cc", borderRadius: 8, padding: 14, alignItems: "center" },
  btnText: { color: "#fff", fontSize: 16, fontWeight: "600" },
  link: { textAlign: "center", color: "#0066cc", marginTop: 8 },
});
```

- [ ] **Step 10: Verify on device**

```bash
npx expo start
```

Scan QR with Expo Go. Verify: app opens to login, sign up works, confirmation email arrives, login navigates to tabs.

- [ ] **Step 11: Commit**

```bash
git init && git add -A
git commit -m "feat: Expo scaffold with Supabase auth"
```

---

### Task 6: Camera screen + Preview screen

**Files:**
- Create: `app/(tabs)/index.tsx` — camera screen
- Create: `app/(tabs)/_layout.tsx` — tab bar
- Create: `app/preview.tsx` — preview + session type picker + submit

**Interfaces:**
- Consumes: `useAuthStore().session.access_token` for API call
- Produces: `POST /api/workouts/photo` call with captured JPEG + session_type
- Navigation: `/(tabs)/index` → `/(tabs)/preview` (stack within tab) → `/workout/[id]`

- [ ] **Step 1: Create `lib/api.ts`**

```typescript
import { useAuthStore } from "./store";

const BASE_URL = process.env.EXPO_PUBLIC_API_URL!;

async function apiFetch(path: string, options: RequestInit = {}) {
  const session = useAuthStore.getState().session;
  const headers: Record<string, string> = {
    ...(options.headers as Record<string, string>),
  };
  if (session?.access_token) {
    headers["Authorization"] = `Bearer ${session.access_token}`;
  }
  const res = await fetch(`${BASE_URL}${path}`, { ...options, headers });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

export const api = {
  postPhoto: (imageUri: string, sessionType: string) => {
    const form = new FormData();
    form.append("image", { uri: imageUri, name: "screen.jpg", type: "image/jpeg" } as any);
    form.append("session_type", sessionType);
    return apiFetch("/api/workouts/photo", { method: "POST", body: form });
  },
  listWorkouts: (sessionType?: string) => {
    const q = sessionType ? `?session_type=${sessionType}` : "";
    return apiFetch(`/api/workouts${q}`);
  },
  getWorkout: (id: number) => apiFetch(`/api/workouts/${id}`),
  getSessionTypes: () => apiFetch("/api/workouts/session-types"),
  analyzeCSV: (fileUri: string, fileName: string, sessionType: string) => {
    const form = new FormData();
    form.append("file", { uri: fileUri, name: fileName, type: "text/csv" } as any);
    form.append("session_type", sessionType);
    return apiFetch("/api/workouts/analyze", { method: "POST", body: form });
  },
  deleteWorkout: (id: number) =>
    apiFetch(`/api/workouts/${id}`, { method: "DELETE" }),
  getCoach: (id: number) =>
    apiFetch(`/api/workouts/${id}/coach`, { method: "POST" }),
};
```

- [ ] **Step 2: Create `app/(tabs)/_layout.tsx`**

```typescript
import { Tabs } from "expo-router";
import { Ionicons } from "@expo/vector-icons";

export default function TabLayout() {
  return (
    <Tabs screenOptions={{ headerShown: false, tabBarActiveTintColor: "#0066cc" }}>
      <Tabs.Screen
        name="index"
        options={{
          title: "Log",
          tabBarIcon: ({ color, size }) => <Ionicons name="camera" size={size} color={color} />,
        }}
      />
      <Tabs.Screen
        name="history"
        options={{
          title: "History",
          tabBarIcon: ({ color, size }) => <Ionicons name="list" size={size} color={color} />,
        }}
      />
    </Tabs>
  );
}
```

- [ ] **Step 3: Create `app/(tabs)/index.tsx` (camera screen)**

```typescript
import { useEffect, useRef, useState } from "react";
import { View, Text, TouchableOpacity, StyleSheet, Alert } from "react-native";
import { CameraView, useCameraPermissions } from "expo-camera";
import { useRouter } from "expo-router";

export default function CameraScreen() {
  const [permission, requestPermission] = useCameraPermissions();
  const cameraRef = useRef<CameraView>(null);
  const router = useRouter();
  const [capturing, setCapturing] = useState(false);

  if (!permission) return <View style={styles.container} />;

  if (!permission.granted) {
    return (
      <View style={styles.container}>
        <Text style={styles.permText}>Camera access is required to log workouts.</Text>
        <TouchableOpacity style={styles.btn} onPress={requestPermission}>
          <Text style={styles.btnText}>Allow camera</Text>
        </TouchableOpacity>
      </View>
    );
  }

  async function takePicture() {
    if (!cameraRef.current || capturing) return;
    setCapturing(true);
    try {
      const photo = await cameraRef.current.takePictureAsync({ quality: 0.85 });
      router.push({ pathname: "/preview", params: { uri: photo!.uri } });
    } catch {
      Alert.alert("Error", "Could not capture photo.");
    } finally {
      setCapturing(false);
    }
  }

  return (
    <View style={styles.container}>
      <CameraView ref={cameraRef} style={styles.camera} facing="back">
        <View style={styles.overlay}>
          <Text style={styles.hint}>Point at the erg summary screen</Text>
          <TouchableOpacity style={styles.shutter} onPress={takePicture} disabled={capturing} />
        </View>
      </CameraView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#000", justifyContent: "center", alignItems: "center" },
  camera: { flex: 1, width: "100%" },
  overlay: { flex: 1, justifyContent: "flex-end", alignItems: "center", paddingBottom: 48 },
  hint: { color: "#fff", fontSize: 16, marginBottom: 24, textShadowColor: "#000", textShadowRadius: 4 },
  shutter: { width: 72, height: 72, borderRadius: 36, backgroundColor: "#fff", borderWidth: 4, borderColor: "#ccc" },
  permText: { color: "#fff", textAlign: "center", marginHorizontal: 32, marginBottom: 16 },
  btn: { backgroundColor: "#0066cc", borderRadius: 8, padding: 14, paddingHorizontal: 24 },
  btnText: { color: "#fff", fontSize: 16, fontWeight: "600" },
});
```

- [ ] **Step 4: Create `app/preview.tsx`**

```typescript
import { useState } from "react";
import { View, Image, Text, TouchableOpacity, StyleSheet, Alert, ActivityIndicator, ScrollView } from "react-native";
import { useLocalSearchParams, useRouter } from "expo-router";
import { useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";
import { Picker } from "@react-native-picker/picker";

export default function PreviewScreen() {
  const { uri } = useLocalSearchParams<{ uri: string }>();
  const router = useRouter();
  const [sessionType, setSessionType] = useState("steady_state");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const { data: sessionTypes = [] } = useQuery({
    queryKey: ["session-types"],
    queryFn: api.getSessionTypes,
  });

  async function submit() {
    setSubmitting(true);
    setError(null);
    try {
      const result = await api.postPhoto(uri, sessionType);
      router.replace({ pathname: "/workout/[id]", params: { id: String(result.workout_id) } });
    } catch (e: any) {
      setError(e.message || "Could not read erg screen — try better lighting or a straighter angle.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Image source={{ uri }} style={styles.preview} resizeMode="contain" />

      <Text style={styles.label}>Session type</Text>
      <View style={styles.pickerWrap}>
        <Picker selectedValue={sessionType} onValueChange={setSessionType}>
          {sessionTypes.map((t: { value: string; label: string }) => (
            <Picker.Item key={t.value} label={t.label} value={t.value} />
          ))}
        </Picker>
      </View>

      {error && <Text style={styles.error}>{error}</Text>}

      <View style={styles.actions}>
        <TouchableOpacity style={styles.secondaryBtn} onPress={() => router.back()} disabled={submitting}>
          <Text style={styles.secondaryText}>Retake</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.primaryBtn} onPress={submit} disabled={submitting}>
          {submitting ? <ActivityIndicator color="#fff" /> : <Text style={styles.primaryText}>Log it</Text>}
        </TouchableOpacity>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { padding: 16, gap: 16 },
  preview: { width: "100%", height: 280, borderRadius: 8, backgroundColor: "#f0f0f0" },
  label: { fontWeight: "600", fontSize: 14 },
  pickerWrap: { borderWidth: 1, borderColor: "#ccc", borderRadius: 8 },
  error: { color: "#cc0000", textAlign: "center" },
  actions: { flexDirection: "row", gap: 12 },
  primaryBtn: { flex: 1, backgroundColor: "#0066cc", borderRadius: 8, padding: 14, alignItems: "center" },
  primaryText: { color: "#fff", fontWeight: "600", fontSize: 16 },
  secondaryBtn: { flex: 1, borderWidth: 1, borderColor: "#ccc", borderRadius: 8, padding: 14, alignItems: "center" },
  secondaryText: { fontSize: 16 },
});
```

- [ ] **Step 5: Add `@react-native-picker/picker`**

```bash
npx expo install @react-native-picker/picker
```

- [ ] **Step 6: Verify on device**

Open Expo Go. Tap Log tab. Verify: camera viewfinder appears, tap shutter captures photo, preview screen shows photo + session type picker, tapping "Log it" calls the backend, navigates to workout detail on success, shows error message inline if backend returns 422.

- [ ] **Step 7: Commit**

```bash
git add lib/api.ts app/\(tabs\)/_layout.tsx app/\(tabs\)/index.tsx app/preview.tsx
git commit -m "feat: camera screen, preview screen, API client"
```

---

### Task 7: History list + Workout detail

**Files:**
- Create: `app/(tabs)/history.tsx`
- Create: `components/WorkoutCard.tsx`
- Create: `app/workout/[id].tsx`

**Interfaces:**
- Consumes: `api.listWorkouts()`, `api.getWorkout(id)`, `api.getCoach(id)`
- Produces: history list with cards; detail view with photo tier vs CSV tier branching on `source`

- [ ] **Step 1: Create `components/WorkoutCard.tsx`**

```typescript
import { View, Text, TouchableOpacity, StyleSheet } from "react-native";

interface Props {
  item: {
    id: number;
    session_label: string;
    uploaded_at: string;
    source: string;
    avg_power?: number;
    overall_score?: number;
    letter?: string;
  };
  onPress: () => void;
}

export function WorkoutCard({ item, onPress }: Props) {
  const date = new Date(item.uploaded_at).toLocaleDateString();
  return (
    <TouchableOpacity style={styles.card} onPress={onPress}>
      <View style={styles.row}>
        <View>
          <Text style={styles.label}>{item.session_label}</Text>
          <Text style={styles.meta}>{date} · {item.source === "photo" ? "📸 photo" : "📊 CSV"}</Text>
        </View>
        <View style={styles.scoreCol}>
          {item.letter && <Text style={styles.letter}>{item.letter}</Text>}
          {item.overall_score != null && <Text style={styles.score}>{item.overall_score.toFixed(0)}</Text>}
        </View>
      </View>
      {item.avg_power != null && (
        <Text style={styles.power}>{item.avg_power.toFixed(0)} W avg</Text>
      )}
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  card: { backgroundColor: "#fff", borderRadius: 10, padding: 16, marginBottom: 10, shadowColor: "#000", shadowOpacity: 0.06, shadowRadius: 6 },
  row: { flexDirection: "row", justifyContent: "space-between", alignItems: "flex-start" },
  label: { fontWeight: "600", fontSize: 16 },
  meta: { color: "#666", fontSize: 13, marginTop: 2 },
  scoreCol: { alignItems: "center" },
  letter: { fontSize: 28, fontWeight: "700", color: "#0066cc" },
  score: { fontSize: 12, color: "#666" },
  power: { color: "#444", fontSize: 14, marginTop: 8 },
});
```

- [ ] **Step 2: Create `app/(tabs)/history.tsx`**

```typescript
import { FlatList, View, Text, StyleSheet, RefreshControl } from "react-native";
import { useRouter } from "expo-router";
import { useQuery } from "@tanstack/react-query";
import { api } from "../../lib/api";
import { WorkoutCard } from "../../components/WorkoutCard";

export default function HistoryScreen() {
  const router = useRouter();
  const { data: workouts = [], isLoading, refetch } = useQuery({
    queryKey: ["workouts"],
    queryFn: () => api.listWorkouts(),
  });

  if (isLoading) {
    return <View style={styles.center}><Text>Loading…</Text></View>;
  }

  return (
    <View style={styles.container}>
      <Text style={styles.heading}>Workout history</Text>
      <FlatList
        data={workouts}
        keyExtractor={(item: any) => String(item.id)}
        renderItem={({ item }) => (
          <WorkoutCard
            item={item}
            onPress={() => router.push({ pathname: "/workout/[id]", params: { id: String(item.id) } })}
          />
        )}
        refreshControl={<RefreshControl refreshing={isLoading} onRefresh={refetch} />}
        ListEmptyComponent={<Text style={styles.empty}>No workouts yet. Log one with the camera!</Text>}
        contentContainerStyle={styles.list}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#f5f5f5", paddingTop: 56 },
  heading: { fontSize: 22, fontWeight: "700", paddingHorizontal: 16, marginBottom: 8 },
  list: { paddingHorizontal: 16, paddingBottom: 32 },
  center: { flex: 1, justifyContent: "center", alignItems: "center" },
  empty: { textAlign: "center", color: "#888", marginTop: 48 },
});
```

- [ ] **Step 3: Create `app/workout/[id].tsx`**

```typescript
import { useState } from "react";
import { ScrollView, View, Text, StyleSheet, TouchableOpacity, ActivityIndicator, Alert } from "react-native";
import { useLocalSearchParams, useRouter } from "expo-router";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../../lib/api";

function StatRow({ label, value }: { label: string; value: string }) {
  return (
    <View style={styles.statRow}>
      <Text style={styles.statLabel}>{label}</Text>
      <Text style={styles.statValue}>{value}</Text>
    </View>
  );
}

export default function WorkoutDetailScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const router = useRouter();
  const queryClient = useQueryClient();
  const [loadingCoach, setLoadingCoach] = useState(false);
  const [coach, setCoach] = useState<any>(null);

  const { data: workout, isLoading } = useQuery({
    queryKey: ["workout", id],
    queryFn: () => api.getWorkout(Number(id)),
  });

  if (isLoading || !workout) {
    return <View style={styles.center}><ActivityIndicator /></View>;
  }

  const summary = workout.summary || {};
  const rating = workout.rating || {};
  const isPhoto = workout.source === "photo";

  async function loadCoach() {
    setLoadingCoach(true);
    try {
      const result = await api.getCoach(Number(id));
      setCoach(result.coach);
    } catch (e: any) {
      Alert.alert("Error", e.message);
    } finally {
      setLoadingCoach(false);
    }
  }

  async function deleteWorkout() {
    Alert.alert("Delete", "Remove this workout?", [
      { text: "Cancel", style: "cancel" },
      {
        text: "Delete", style: "destructive", onPress: async () => {
          await api.deleteWorkout(Number(id));
          queryClient.invalidateQueries({ queryKey: ["workouts"] });
          router.back();
        },
      },
    ]);
  }

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <TouchableOpacity onPress={() => router.back()} style={styles.back}>
        <Text style={styles.backText}>← Back</Text>
      </TouchableOpacity>

      <View style={styles.header}>
        <Text style={styles.title}>{workout.session_label}</Text>
        <Text style={styles.meta}>
          {new Date(workout.uploaded_at).toLocaleDateString()} · {isPhoto ? "📸 photo" : "📊 CSV"}
        </Text>
      </View>

      <View style={styles.scoreCard}>
        <Text style={styles.scoreLetter}>{rating.letter || "—"}</Text>
        <Text style={styles.scoreNum}>{rating.overall?.toFixed(0) ?? "—"}</Text>
        {isPhoto && <Text style={styles.scoreNote}>Effort score only — upload CSV for full analysis</Text>}
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>Summary</Text>
        {summary.meters != null && <StatRow label="Distance" value={`${summary.meters} m`} />}
        {summary.elapsed_time && <StatRow label="Time" value={summary.elapsed_time} />}
        {summary.avg_split && <StatRow label="Avg split" value={`${summary.avg_split} /500m`} />}
        {summary.avg_power != null && <StatRow label="Avg power" value={`${summary.avg_power} W`} />}
        {summary.stroke_rate != null && <StatRow label="Stroke rate" value={`${summary.stroke_rate} spm`} />}
      </View>

      {!isPhoto && workout.comparison?.has_prior && (
        <View style={styles.card}>
          <Text style={styles.cardTitle}>vs last session</Text>
          {(workout.comparison.vs_previous || []).map((d: any) => (
            <StatRow key={d.label} label={d.label} value={`${d.delta > 0 ? "+" : ""}${d.delta?.toFixed(1)} ${d.unit || ""}`} />
          ))}
        </View>
      )}

      <View style={styles.card}>
        <View style={styles.coachHeader}>
          <Text style={styles.cardTitle}>Coaching</Text>
          <TouchableOpacity onPress={loadCoach} disabled={loadingCoach}>
            {loadingCoach ? <ActivityIndicator /> : <Text style={styles.coachBtn}>Get coaching</Text>}
          </TouchableOpacity>
        </View>
        {coach && (
          <>
            <Text style={styles.headline}>{coach.headline}</Text>
            {coach.went_well?.map((b: string, i: number) => <Text key={i} style={styles.bullet}>✓ {b}</Text>)}
            {coach.work_on?.map((b: string, i: number) => <Text key={i} style={styles.bullet}>→ {b}</Text>)}
            {coach.next_session && <Text style={styles.next}>Next: {coach.next_session}</Text>}
          </>
        )}
      </View>

      <TouchableOpacity style={styles.deleteBtn} onPress={deleteWorkout}>
        <Text style={styles.deleteText}>Delete workout</Text>
      </TouchableOpacity>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#f5f5f5" },
  content: { padding: 16, paddingBottom: 48 },
  center: { flex: 1, justifyContent: "center", alignItems: "center" },
  back: { marginTop: 48, marginBottom: 8 },
  backText: { color: "#0066cc", fontSize: 16 },
  header: { marginBottom: 16 },
  title: { fontSize: 22, fontWeight: "700" },
  meta: { color: "#666", marginTop: 4 },
  scoreCard: { backgroundColor: "#fff", borderRadius: 10, padding: 20, alignItems: "center", marginBottom: 12 },
  scoreLetter: { fontSize: 56, fontWeight: "700", color: "#0066cc" },
  scoreNum: { fontSize: 20, color: "#444" },
  scoreNote: { fontSize: 12, color: "#888", marginTop: 8, textAlign: "center" },
  card: { backgroundColor: "#fff", borderRadius: 10, padding: 16, marginBottom: 12 },
  cardTitle: { fontWeight: "600", fontSize: 16, marginBottom: 8 },
  statRow: { flexDirection: "row", justifyContent: "space-between", paddingVertical: 4 },
  statLabel: { color: "#555" },
  statValue: { fontWeight: "500" },
  coachHeader: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", marginBottom: 8 },
  coachBtn: { color: "#0066cc", fontWeight: "600" },
  headline: { fontWeight: "600", marginBottom: 8 },
  bullet: { color: "#333", marginBottom: 4 },
  next: { marginTop: 8, fontStyle: "italic", color: "#555" },
  deleteBtn: { padding: 14, alignItems: "center" },
  deleteText: { color: "#cc0000" },
});
```

- [ ] **Step 4: Verify on device**

Open History tab — workout list appears. Tap a card — detail view opens. Photo workouts show "Effort score only" note and hide comparison/chart. CSV workouts show full detail. Coaching button calls backend and renders response.

- [ ] **Step 5: Commit**

```bash
git add components/WorkoutCard.tsx app/\(tabs\)/history.tsx app/workout/
git commit -m "feat: history list and workout detail screen"
```

---

### Task 8: CSV upload modal

**Files:**
- Create: `app/upload.tsx`
- Modify: `app/(tabs)/history.tsx` — add upload button in header

**Interfaces:**
- Consumes: `api.analyzeCSV(fileUri, fileName, sessionType)` → `WorkoutAnalyzeResponse`
- Produces: file picker → POST to `/api/workouts/analyze` → navigate to workout detail

- [ ] **Step 1: Create `app/upload.tsx`**

```typescript
import { useState } from "react";
import { View, Text, TouchableOpacity, StyleSheet, ActivityIndicator, Alert } from "react-native";
import * as DocumentPicker from "expo-document-picker";
import { useRouter } from "expo-router";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Picker } from "@react-native-picker/picker";
import { api } from "../lib/api";

export default function UploadScreen() {
  const router = useRouter();
  const queryClient = useQueryClient();
  const [sessionType, setSessionType] = useState("steady_state");
  const [file, setFile] = useState<{ uri: string; name: string } | null>(null);
  const [uploading, setUploading] = useState(false);

  const { data: sessionTypes = [] } = useQuery({
    queryKey: ["session-types"],
    queryFn: api.getSessionTypes,
  });

  async function pickFile() {
    const result = await DocumentPicker.getDocumentAsync({ type: "text/comma-separated-values" });
    if (!result.canceled && result.assets[0]) {
      setFile({ uri: result.assets[0].uri, name: result.assets[0].name });
    }
  }

  async function upload() {
    if (!file) return;
    setUploading(true);
    try {
      const result = await api.analyzeCSV(file.uri, file.name, sessionType);
      queryClient.invalidateQueries({ queryKey: ["workouts"] });
      router.replace({ pathname: "/workout/[id]", params: { id: String(result.workout_id) } });
    } catch (e: any) {
      Alert.alert("Upload failed", e.message);
    } finally {
      setUploading(false);
    }
  }

  return (
    <View style={styles.container}>
      <TouchableOpacity onPress={() => router.back()} style={styles.back}>
        <Text style={styles.backText}>✕ Cancel</Text>
      </TouchableOpacity>
      <Text style={styles.title}>Upload CSV</Text>
      <Text style={styles.hint}>Export a single workout as CSV from Concept2 Logbook (stroke data required).</Text>

      <Text style={styles.label}>Session type</Text>
      <View style={styles.pickerWrap}>
        <Picker selectedValue={sessionType} onValueChange={setSessionType}>
          {sessionTypes.map((t: { value: string; label: string }) => (
            <Picker.Item key={t.value} label={t.label} value={t.value} />
          ))}
        </Picker>
      </View>

      <TouchableOpacity style={styles.filePicker} onPress={pickFile}>
        <Text style={styles.filePickerText}>{file ? file.name : "Choose CSV file"}</Text>
      </TouchableOpacity>

      <TouchableOpacity style={[styles.btn, !file && styles.btnDisabled]} onPress={upload} disabled={!file || uploading}>
        {uploading ? <ActivityIndicator color="#fff" /> : <Text style={styles.btnText}>Analyze workout</Text>}
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, padding: 24, paddingTop: 56, backgroundColor: "#fff" },
  back: { marginBottom: 16 },
  backText: { color: "#0066cc" },
  title: { fontSize: 22, fontWeight: "700", marginBottom: 8 },
  hint: { color: "#666", marginBottom: 24, lineHeight: 20 },
  label: { fontWeight: "600", marginBottom: 6 },
  pickerWrap: { borderWidth: 1, borderColor: "#ccc", borderRadius: 8, marginBottom: 16 },
  filePicker: { borderWidth: 1, borderColor: "#ccc", borderRadius: 8, padding: 14, marginBottom: 16, alignItems: "center" },
  filePickerText: { color: "#555" },
  btn: { backgroundColor: "#0066cc", borderRadius: 8, padding: 14, alignItems: "center" },
  btnDisabled: { backgroundColor: "#aaa" },
  btnText: { color: "#fff", fontWeight: "600", fontSize: 16 },
});
```

- [ ] **Step 2: Add upload button to history header**

In `app/(tabs)/history.tsx`, import `useRouter` and add a header button:

```typescript
// Add at the top of HistoryScreen's return:
<View style={styles.header}>
  <Text style={styles.heading}>Workout history</Text>
  <TouchableOpacity onPress={() => router.push("/upload")}>
    <Text style={styles.uploadBtn}>Upload CSV</Text>
  </TouchableOpacity>
</View>
```

Add to styles:
```typescript
header: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", paddingHorizontal: 16, marginBottom: 8 },
uploadBtn: { color: "#0066cc", fontSize: 14 },
```

And remove `paddingHorizontal: 16` from the `heading` style (it moves to `header`).

- [ ] **Step 3: Verify on device**

Tap "Upload CSV" in history header. File picker opens — select a Concept2 CSV. Tap "Analyze workout". Full analysis result (power curve, score breakdown) appears in workout detail. Workout appears in history with `source="csv"`.

- [ ] **Step 4: Commit**

```bash
git add app/upload.tsx app/\(tabs\)/history.tsx
git commit -m "feat: CSV upload modal"
```

---

## Environment Variables Required

### Backend (Render)
```
DATABASE_URL=postgresql://postgres:[password]@db.[project].supabase.co:5432/postgres
SUPABASE_JWT_SECRET=<from Supabase dashboard: Settings > API > JWT Secret>
GEMINI_API_KEY=<existing>
```

### Mobile (.env.local)
```
EXPO_PUBLIC_SUPABASE_URL=https://[project].supabase.co
EXPO_PUBLIC_SUPABASE_ANON_KEY=<from Supabase dashboard: Settings > API > anon key>
EXPO_PUBLIC_API_URL=https://[your-render-app].onrender.com
```
