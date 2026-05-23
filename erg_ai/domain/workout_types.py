"""User-selected workout session types."""

from enum import Enum
from typing import Dict


class SessionType(str, Enum):
    STEADY_STATE = "steady_state"
    THRESHOLD = "threshold"
    INTERVALS = "intervals"
    RACE_TEST = "race_test"
    RECOVERY = "recovery"

    @classmethod
    def from_str(cls, value: str) -> "SessionType":
        try:
            return cls(value.strip().lower())
        except ValueError as exc:
            valid = ", ".join(s.value for s in cls)
            raise ValueError(f"Invalid session_type '{value}'. Must be one of: {valid}") from exc


SESSION_TYPE_LABELS: Dict[SessionType, str] = {
    SessionType.STEADY_STATE: "Steady State",
    SessionType.THRESHOLD: "Threshold / UT1",
    SessionType.INTERVALS: "Intervals / VO2",
    SessionType.RACE_TEST: "Race / Test Piece",
    SessionType.RECOVERY: "Recovery",
}


def session_type_labels() -> Dict[str, str]:
    return {st.value: SESSION_TYPE_LABELS[st] for st in SessionType}
