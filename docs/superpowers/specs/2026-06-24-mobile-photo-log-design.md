# Mobile Photo Log — Design Spec
**Date:** 2026-06-24
**Status:** Approved

## Overview

Reimagine erg-ai as a phone-first app. The primary UX is: open app → point phone at Concept2 RowErg summary screen → tap shutter → workout is logged automatically. A secondary CSV upload path is preserved for athletes who want full stroke-level analysis.

**Scope:** Concept2 RowErg only. iOS and Android via Expo.

---

## Architecture

```
iPhone (Expo React Native)
  ├── expo-router           — tab/stack navigation
  ├── expo-camera           — live viewfinder + capture
  ├── @supabase/supabase-js — auth (email + password)
  └── fetch → FastAPI on Render (JWT in Authorization header)

FastAPI (Render)
  ├── Supabase JWT middleware  — verifies token, extracts user_id
  ├── POST /api/workouts/photo   ← NEW
  ├── POST /api/workouts/analyze ← existing CSV path, unchanged
  └── all existing workout endpoints (list, detail, coach, compare, delete)
  └── SQLAlchemy → Supabase Postgres

Supabase (free tier)
  ├── Auth — email/password sign-up + login
  └── Postgres — all workout data, replaces SQLite

Gemini API
  ├── Vision — photo OCR (new)
  └── Text  — coaching (existing)
```

The backend stays deployed on Render. Supabase replaces SQLite and provides auth. The Expo app is a new repository. Existing API endpoints are untouched except for the JWT middleware and DB URL swap.

---

## Screens & Navigation

```
Auth stack (shown before tabs if not signed in)
  ├── /login       — email + password, link to sign up
  └── /signup      — email + password + confirm

Tab 1: Log  (default tab, camera icon)
  └── /log/camera  — live viewfinder, shutter button
        └── /log/preview  — confirm photo or retake; session type picker; "Log it" button
              └── /workout/[id]  — auto-navigate after processing

Tab 2: History  (list icon)
  └── /history     — scrollable list, filterable by session type
        └── /workout/[id]  — detail view

Modal (accessible from history header)
  └── /upload      — session type picker + CSV file picker
```

**The camera screen is the home screen.** It is the first tab and the default landing after login.

**Workout detail** renders two tiers based on `source`:
- `photo`: summary stats (meters, time, avg split, avg watts, stroke rate), simplified score, coaching button
- `csv`: full existing view — power curve, stroke quality breakdown, interval detection, comparison

CSV upload is a secondary escape hatch, not primary navigation.

---

## Data Flow — Photo Log

```
1. User taps shutter
2. expo-camera captures JPEG; app shows preview screen
3. User taps "Log it"
4. App POSTs multipart to FastAPI:
     POST /api/workouts/photo
     Authorization: Bearer <supabase-jwt>
     Body: { image: <jpeg bytes>, session_type: string }

5. JWT middleware verifies token → extracts user_id

6. photo_service.py:
   a. Encode image as base64
   b. Call Gemini Vision:
      "This is a Concept2 RowErg summary screen.
       Extract these fields and return JSON only:
       { meters, elapsed_time, avg_split, avg_watts, stroke_rate }
       If a field is not visible, set it to null."
   c. Parse + validate JSON response
   d. Raise HTTP 422 if all fields are null

7. Create Workout row in Postgres:
     source="photo", user_id=<jwt user_id>,
     summary_json=<extracted stats>,
     rating_json=<stats-only rating>

8. Return WorkoutAnalyzeResponse → app navigates to /workout/[id]
```

---

## Backend Changes

### DB migration (additive)
- Add `workouts.source` column: `VARCHAR`, values `"photo"` or `"csv"`, default `"csv"`
- Existing rows backfilled to `"csv"` — no data loss

### New files
- `erg_ai/services/photo_service.py` — Gemini Vision call + field extraction
- `erg_ai/middleware/auth.py` — Supabase JWT verification via `python-jose`

### Modified files
- `erg_ai/config.py` — add `SUPABASE_URL`, `SUPABASE_JWT_SECRET` env vars
- `erg_ai/db/session.py` — swap SQLite URL → Supabase Postgres URL
- `erg_ai/main.py` — register auth middleware
- `erg_ai/api/workouts.py` — replace `user_id: str = "local"` with `user_id` from JWT dependency
- `erg_ai/services/rating_service.py` — skip stroke-quality and drift dimensions when `source="photo"`
- `requirements.txt` — add `python-jose[cryptography]`, `psycopg2-binary`

### New endpoint
```
POST /api/workouts/photo
  Auth: required (JWT)
  Body: multipart — image (JPEG), session_type (string)
  Returns: WorkoutAnalyzeResponse (same schema as CSV analyze)
  Errors:
    422 — Gemini could not extract fields ("bad lighting or angle")
    401 — missing or invalid JWT
```

---

## Mobile App (New Repo)

**Stack:**
- Expo SDK 52+ with expo-router v4
- expo-camera for viewfinder + capture
- @supabase/supabase-js for auth
- React Query for API state + caching
- Zustand for auth session store

**Key implementation notes:**
- Camera screen requests permission on first open; if denied, shows instructions to enable in Settings
- Preview screen keeps the captured JPEG in memory (not saved to camera roll) until "Log it" is tapped
- On submit, photo stays visible in preview with a loading overlay; user cannot double-submit
- On 422, show inline error with retry option — do not navigate away, preserve the photo
- On auth expiry (401), clear session store and redirect to /login

---

## Error Handling

| Scenario | Backend | Mobile |
|---|---|---|
| Gemini can't read screen | 422 + message | Inline error on preview, photo preserved |
| All fields null (blank photo) | 422 | Same as above |
| Network offline | — | Inline error, no navigation |
| JWT expired | 401 | Clear session, redirect to login |
| Gemini API down | 503 | "Service temporarily unavailable, try CSV upload" |

---

## What Is Not In Scope

- Android-specific camera quirks beyond expo-camera defaults
- Multi-erg support (BikeErg, SkiErg)
- Interval-by-interval photo capture
- Social / sharing features
- Push notifications
- App Store submission (Expo Go is sufficient for initial use)
