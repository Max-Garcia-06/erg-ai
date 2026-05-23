# Training Partner

A local-first rowing erg training partner: upload per-workout CSV exports from Concept2 Logbook, build workout history, get **type-aware scores** (steady state, threshold, intervals, race/test, recovery), and optional AI coaching.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Train ML models (optional but recommended)
python features_and_labels.py
python train_models.py

# Run the app
uvicorn erg_ai.main:app --reload
```

Open [http://localhost:8000](http://localhost:8000).

**Workout tabs:** Open multiple workouts from history — each appears in a tab bar so you can switch without losing context. **Comparisons:** On each workout detail, see deltas vs your last same-type session and vs the average of your prior five.

## CSV requirements

- Export **one workout at a time** from Concept2 Logbook with stroke-level data.
- Must include a **Watts** (or Power) column and at least ~50 stroke rows.
- **Bulk year CSV exports** are summary-only and are rejected with a clear error.

## Session types & scoring

Before analyzing, pick what you intended to do. The app scores against a rubric for that type (weights in `config.yaml` → `session_types`):

| Type | Focus |
|------|--------|
| Steady State | Consistency, low drift, technique — **split SS pieces** (e.g. 3×20′) are scored per work segment, not penalized as VO2 intervals |
| Threshold / UT1 | Sustained effort, pacing control |
| Intervals / VO2 | Work/rest structure, repeatability |
| Race / Test Piece | Peak output, pacing strategy |
| Recovery | Technique; low power is not penalized |

You can change the session type on a saved workout and **re-score** without re-uploading.

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/workouts/session-types` | List session types |
| `POST` | `/api/workouts/analyze` | Upload CSV + `session_type` form field |
| `GET` | `/api/workouts` | History (`?session_type=` filter) |
| `GET` | `/api/workouts/{id}` | Workout detail + chart series |
| `PATCH` | `/api/workouts/{id}` | Change `session_type`, re-score |
| `POST` | `/api/workouts/{id}/coach` | Template or Gemini coaching |
| `DELETE` | `/api/workouts/{id}` | Remove workout |

Legacy: `POST /analyze` still works with optional `session_type`.

## Configuration

- `config.yaml` — ML features, stroke quality ranges, **session type rubrics**
- `.env` — optional `GEMINI_API_KEY`, `DATABASE_URL`

Data is stored in `data/erg.db` and uploaded CSVs in `data/uploads/`.

## Coaching

Without `GEMINI_API_KEY`, coaching uses deterministic templates from your rating and focus areas. With a key set, Gemini generates structured feedback (cached per workout).

## Future: multi-user & Logbook sync

The schema uses `user_id` (default `local`) so accounts and Concept2 Logbook OAuth can be added later without changing the analysis pipeline.

## Development

```bash
pytest tests/ -v
```

## Project layout

```
erg_ai/           # Application package
  api/            # REST routes
  db/             # SQLAlchemy models
  domain/         # Session types
  services/       # Ingest, analysis, rating, coach
  clients/        # Gemini (optional)
static/           # Web UI
infer_models.py   # ML inference (unchanged)
features_and_labels.py
train_models.py
```
