"""Concept2 / Logbook CSV ingestion and validation."""

import io
from typing import Tuple

import pandas as pd
from fastapi import HTTPException

from erg_ai.config import get_config

_COLUMN_ALIASES = {
    "time (s)": "time",
    "time (seconds)": "time",
    "time(s)": "time",
    "pace (sec/500m)": "pace",
    "pace (seconds)": "pace",
    "power": "watts",
    "power (watts)": "watts",
    "stroke rate": "stroke_rate",
    "stroke-rate": "stroke_rate",
    "spm": "stroke_rate",
    "heart rate": "heart_rate",
    "hr": "heart_rate",
    "bpm": "heart_rate",
    "distance (meters)": "distance",
    "distance (m)": "distance",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    df.rename(columns=_COLUMN_ALIASES, inplace=True)
    for col in ["watts", "pace", "stroke_rate", "heart_rate", "time", "distance"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "watts" in df.columns:
        df = df.dropna(subset=["watts"])
    return df


def parse_csv_bytes(contents: bytes, filename: str) -> pd.DataFrame:
    cfg = get_config()
    max_bytes = cfg.get("max_file_size_mb", 10) * 1024 * 1024
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {cfg.get('max_file_size_mb', 10)}MB.",
        )

    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {exc}") from exc

    if df.empty:
        raise HTTPException(
            status_code=400,
            detail="CSV file is empty. Export a single workout with stroke data from Concept2 Logbook.",
        )

    df = normalize_columns(df)

    if "watts" not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=(
                "Missing required watts/power column. Bulk Logbook year exports are summary-only — "
                f"export this workout individually. Found columns: {list(df.columns)}"
            ),
        )

    min_rows = cfg.get("min_file_length", 50)
    if len(df) < min_rows:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Only {len(df)} stroke rows found (minimum {min_rows}). "
                "Use a per-workout CSV export with full stroke data."
            ),
        )

    return df


def derive_duration_sec(df: pd.DataFrame) -> float | None:
    if "time" in df.columns and df["time"].notna().any():
        return float(df["time"].max() - df["time"].min())
    return float(len(df))


def chart_series(df: pd.DataFrame, max_points: int = 500) -> dict:
    """Downsample series for frontend charts."""
    n = len(df)
    step = max(1, n // max_points)

    def sample(col: str):
        if col not in df.columns:
            return []
        vals = df[col].iloc[::step].tolist()
        return [None if (v != v) else float(v) for v in vals]

    time_vals = sample("time")
    if not time_vals:
        time_vals = list(range(0, n, step))

    return {
        "time": time_vals,
        "watts": sample("watts"),
        "pace": sample("pace"),
        "stroke_rate": sample("stroke_rate"),
        "heart_rate": sample("heart_rate"),
    }
