from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator

# The canonical set of states the AI liveness engine may report.
# Only REAL_LOCKED is accepted as proof of genuine human presence.
VALID_LIVENESS_STATES = {"NEW", "CHECKING", "REAL_LOCKED", "HOLD", "SPOOF", "RESET", "UNCERTAIN"}


class AttendanceEventIn(BaseModel):
    student_id: Optional[str] = Field(None, description="Student ID/Roll number string, e.g. '23UAM090'")
    session_id: int = Field(..., description="Target session database ID")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Recognition confidence (cosine similarity)")
    # Primary liveness gate — must be REAL_LOCKED to be admitted.
    liveness_state: str = Field(
        ...,
        description="Temporal liveness state from AI engine. Must be 'REAL_LOCKED' to admit attendance.",
    )
    # Optional diagnostic metadata — never used as a gate.
    liveness_score: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Raw anti-spoof probability (diagnostic metadata only, not used as gate).",
    )
    source: str = Field(default="AI_CAMERA", description="Source of event: AI_CAMERA, LAPTOP_CAMERA, MANUAL")
    timestamp: Optional[Union[datetime, str]] = Field(None, description="ISO timestamp of observation")

    @field_validator("liveness_state")
    @classmethod
    def validate_liveness_state(cls, v: str) -> str:
        state = v.upper()
        if state not in VALID_LIVENESS_STATES:
            raise ValueError(
                f"Unknown liveness_state '{v}'. Valid states: {sorted(VALID_LIVENESS_STATES)}"
            )
        return state


class AttendanceEventOut(BaseModel):
    success: bool
    status: str = Field(..., description="PRESENT, DUPLICATE, BLOCKED, or REJECTED")
    message: str
    attendance_id: Optional[int] = None
    student_id: Optional[str] = None
    student_name: Optional[str] = None


class AttendanceRecordResponse(BaseModel):
    id: int
    session_id: int
    student_id: str
    student_name: str
    timestamp: datetime
    status: str
    confidence: Optional[float] = None
    liveness_score: Optional[float] = None
    liveness_state: Optional[str] = None
    source: str

    model_config = {"from_attributes": True}
