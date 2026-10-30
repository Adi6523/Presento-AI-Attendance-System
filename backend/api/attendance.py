from __future__ import annotations

from typing import List, Optional
from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import User
from ..schemas.attendance import AttendanceEventIn, AttendanceEventOut, AttendanceRecordResponse
from ..services.attendance_service import get_attendance_history, process_attendance_event
from .dependencies import get_current_user, verify_ai_api_key

router = APIRouter(prefix="/api/attendance", tags=["Attendance"])


@router.post(
    "/events",
    response_model=AttendanceEventOut,
    status_code=status.HTTP_200_OK,
    summary="Process AI Attendance Event",
    description=(
        "Primary contract endpoint for the AI Vision engine. "
        "Requires Bearer `PRESENTO_AI_API_KEY` (not a frontend JWT). "
        "Executes 9-step validation pipeline — including liveness_state == 'REAL_LOCKED' — "
        "and returns PRESENT, DUPLICATE, BLOCKED, or REJECTED."
    ),
)
def handle_attendance_event(
    event: AttendanceEventIn,
    db: Session = Depends(get_db),
    _: None = Depends(verify_ai_api_key),
):
    """
    Validates AI observation:
    - Enrolled student
    - Active student
    - Active session (ACTIVE status)
    - Student enrolled in the session's class
    - Confidence >= threshold (default 0.48)
    - liveness_state == 'REAL_LOCKED'  ← primary liveness gate
    - Duplicate protection via DB UNIQUE constraint
    """
    return process_attendance_event(db, event)


@router.get(
    "/history",
    response_model=List[AttendanceRecordResponse],
    status_code=status.HTTP_200_OK,
    summary="Attendance History",
    description="Retrieve historical attendance records with filters for date, class, student, or status.",
)
def query_attendance_history(
    date: Optional[str] = Query(None, description="Filter by date YYYY-MM-DD"),
    class_id: Optional[int] = Query(None, description="Filter by class ID"),
    student_id: Optional[str] = Query(None, description="Filter by student roll number / ID"),
    status: Optional[str] = Query(None, description="Filter by status: PRESENT, ABSENT, BLOCKED, REJECTED"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return get_attendance_history(db, date=date, class_id=class_id, student_id=student_id, status=status)
