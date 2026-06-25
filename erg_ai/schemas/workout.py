"""Pydantic API schemas."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SessionTypeInfo(BaseModel):
    value: str
    label: str


class WorkoutListItem(BaseModel):
    id: int
    filename: str
    uploaded_at: datetime
    session_type: str
    session_label: str
    title: str
    source: str = "csv"
    avg_power: Optional[float] = None
    overall_score: Optional[float] = None
    letter: Optional[str] = None
    focus_areas: List[Dict[str, Any]] = Field(default_factory=list)


class WorkoutAnalyzeResponse(BaseModel):
    workout_id: int
    filename: str
    session_type: str
    session_label: str
    summary: Dict[str, Any]
    metrics: Dict[str, Any]
    rating: Dict[str, Any]
    chart_series: Dict[str, Any]


class WorkoutComparison(BaseModel):
    session_type: str
    session_label: str
    has_prior: bool = False
    previous: Optional[Dict[str, Any]] = None
    prior_same_type: List[Dict[str, Any]] = Field(default_factory=list)
    last_5_average: Dict[str, Any] = Field(default_factory=dict)
    vs_previous: List[Dict[str, Any]] = Field(default_factory=list)
    vs_last_5_average: List[Dict[str, Any]] = Field(default_factory=list)


class WorkoutDetailResponse(BaseModel):
    id: int
    filename: str
    uploaded_at: datetime
    session_type: str
    session_label: str
    title: str
    notes: Optional[str] = None
    source: str = "csv"
    detected_structure: str
    duration_sec: Optional[float] = None
    summary: Dict[str, Any]
    metrics: Dict[str, Any]
    rating: Dict[str, Any]
    chart_series: Dict[str, Any]
    coach: Optional[Dict[str, Any]] = None
    comparison: Optional[WorkoutComparison] = None


class WorkoutPatchRequest(BaseModel):
    session_type: Optional[str] = None
    title: Optional[str] = None
    notes: Optional[str] = None


class CoachResponse(BaseModel):
    workout_id: int
    coach: Dict[str, Any]
