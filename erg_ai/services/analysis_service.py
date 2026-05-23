"""Workout metrics and ML analysis."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from erg_ai.config import get_config
from infer_models import InferenceEngine

logger = logging.getLogger(__name__)

_engine: Optional[InferenceEngine] = None


def get_inference_engine() -> Optional[InferenceEngine]:
    global _engine
    if _engine is None:
        try:
            _engine = InferenceEngine()
            logger.info("Inference engine loaded")
        except Exception as exc:
            logger.error("Failed to load inference engine: %s", exc)
    return _engine


def compute_stroke_quality_components(df: pd.DataFrame) -> Dict[str, float]:
    cfg = get_config()
    watts_arr = df["watts"].values

    rolling_std = float(
        pd.Series(watts_arr).rolling(10, min_periods=1).std().dropna().mean()
    )
    watts_jitter = float(pd.Series(watts_arr).diff().abs().mean())

    sr_mean = (
        float(df["stroke_rate"].dropna().mean())
        if "stroke_rate" in df.columns and df["stroke_rate"].notna().any()
        else None
    )
    avg_watts = float(np.nanmean(watts_arr))
    efficiency = avg_watts / sr_mean if (sr_mean and sr_mean > 0) else avg_watts / 150.0

    eff_range = cfg.get("efficiency_range", [0.8, 2.5])
    cons_range = cfg.get("consistency_range", [5, 50])
    smooth_range = cfg.get("smoothness_range", [5, 50])

    norm_eff = float(
        np.clip((efficiency - eff_range[0]) / (eff_range[1] - eff_range[0]), 0, 1)
    )
    norm_cons = float(
        np.clip(1 - (rolling_std - cons_range[0]) / (cons_range[1] - cons_range[0]), 0, 1)
    )
    norm_smooth = float(
        np.clip(
            1 - (watts_jitter - smooth_range[0]) / (smooth_range[1] - smooth_range[0]),
            0,
            1,
        )
    )

    eff_w = cfg.get("efficiency_weight", 0.4)
    cons_w = cfg.get("consistency_weight", 0.35)
    smooth_w = cfg.get("smoothness_weight", 0.25)
    overall = (eff_w * norm_eff + cons_w * norm_cons + smooth_w * norm_smooth) * 100

    return {
        "overall": round(overall, 1),
        "efficiency": round(norm_eff * 100, 1),
        "consistency": round(norm_cons * 100, 1),
        "drift": round(norm_smooth * 100, 1),
    }


def detect_structure(interval_pairs: List) -> str:
    if len(interval_pairs) >= 1:
        return "interval_pattern"
    return "continuous"


def compute_interval_segment_stats(
    df: pd.DataFrame, interval_pairs: List[List[int]]
) -> Dict[str, Any]:
    """Per work-segment stats for split steady-state (and interval) scoring."""
    segments: List[Dict[str, Any]] = []
    for pair in interval_pairs:
        if len(pair) < 2:
            continue
        start, end = int(pair[0]), int(pair[1])
        end = min(end, len(df) - 1)
        start = max(0, start)
        if end <= start:
            continue

        seg = df.iloc[start : end + 1]["watts"].dropna()
        if len(seg) < 2:
            continue

        w = seg.values
        window = max(1, len(w) // 10)
        seg_drift = float(w[-window:].mean() - w[:window].mean())

        segments.append(
            {
                "start": start,
                "end": end,
                "avg_watts": round(float(np.mean(w)), 1),
                "std_watts": round(float(np.std(w)), 1),
                "drift_watts": round(seg_drift, 1),
                "row_count": len(seg),
            }
        )

    repeatability_cv: Optional[float] = None
    if len(segments) >= 2:
        avgs = [s["avg_watts"] for s in segments]
        mean_w = float(np.mean(avgs))
        if mean_w > 0:
            repeatability_cv = round(float(np.std(avgs) / mean_w), 4)

    return {
        "interval_segments": segments,
        "segment_count": len(segments),
        "segment_avg_std_watts": (
            round(float(np.mean([s["std_watts"] for s in segments])), 1)
            if segments
            else None
        ),
        "segment_avg_drift_watts": (
            round(float(np.mean([abs(s["drift_watts"]) for s in segments])), 1)
            if segments
            else None
        ),
        "segment_repeatability_cv": repeatability_cv,
        "steady_state_format": (
            "split_intervals" if len(segments) >= 2 else "continuous"
        ),
    }


def analyze_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """Run ML + rule-based analysis; works without ML in degraded mode."""
    cfg = get_config()
    engine = get_inference_engine()

    watts_vals = df["watts"].dropna()
    avg_power = float(watts_vals.mean())
    watts_std = float(watts_vals.std()) if len(watts_vals) > 1 else 0.0

    window = max(1, len(watts_vals) // 10)
    drift = float(watts_vals.iloc[-window:].mean() - watts_vals.iloc[:window].mean())

    avg_split = (
        float(df["pace"].dropna().mean())
        if "pace" in df.columns and df["pace"].notna().any()
        else None
    )

    sq_components = compute_stroke_quality_components(df)

    interval_pairs: List[List[int]] = []
    anomaly_indices: List[int] = []
    sq_pred: Optional[float] = None
    ml_available = False

    if engine is not None:
        ml_results = engine.analyze(df)
        ml_available = "error" not in ml_results

        interval_result = ml_results.get("intervals", {})
        if interval_result.get("available"):
            interval_pairs = [list(p) for p in interval_result.get("intervals", [])]

        anomaly_result = ml_results.get("anomalies", {})
        if anomaly_result.get("available"):
            anomaly_indices = anomaly_result.get("anomalies", [])

        sq_result = ml_results.get("stroke_quality", {})
        if sq_result.get("available"):
            sq_pred = sq_result.get("score")

    detected_structure = detect_structure(interval_pairs)
    segment_stats = compute_interval_segment_stats(df, interval_pairs)

    return {
        "avg_power": round(avg_power, 1),
        "avg_split": round(avg_split, 1) if avg_split is not None else None,
        "consistency": round(watts_std, 1),
        "drift": round(drift, 1),
        "interval_report": (
            f"{len(interval_pairs)} interval(s) detected"
            if interval_pairs
            else "No intervals detected"
        ),
        "intervals": interval_pairs,
        "anomalies": anomaly_indices,
        "stroke_quality": sq_components,
        "sq_pred": round(sq_pred, 1) if sq_pred is not None else None,
        "detected_structure": detected_structure,
        "steady_state_format": segment_stats["steady_state_format"],
        **segment_stats,
        "ml_available": ml_available,
        "row_count": len(df),
    }
