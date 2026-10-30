from __future__ import annotations

from typing import List, Optional
from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import SessionModel, User
from ..schemas.attendance import AttendanceRecordResponse
from ..schemas.session import SessionCreate, SessionResponse
from ..services.attendance_service import get_session_attendance
from ..services.session_service import (
    cancel_session,
    close_session,
    create_session,
    get_session_by_id,
    list_sessions,
    start_session,
)
from .dependencies import get_current_user, get_current_user_or_ai_key, require_role

router = APIRouter(prefix="/api/sessions", tags=["Sessions"])


def _format_session_response(s: SessionModel) -> SessionResponse:
    return SessionResponse(
        id=s.id,
        class_id=s.class_id,
        teacher_id=s.teacher_id,
        date=s.date,
        start_time=s.start_time,
        end_time=s.end_time,
        status=s.status,
        created_at=s.created_at,
        class_name=s.class_model.name if s.class_model else None,
        subject=s.class_model.subject if s.class_model else None,
    )


@router.get(
    "",
    response_model=List[SessionResponse],
    status_code=status.HTTP_200_OK,
    summary="List Sessions",
    description="Retrieve classroom sessions with optional filtering by class, status, or date.",
)
def get_sessions(
    class_id: Optional[int] = Query(None, description="Filter by class ID"),
    status: Optional[str] = Query(None, description="Filter by status (SCHEDULED, ACTIVE, COMPLETED, CANCELLED)"),
    date: Optional[str] = Query(None, description="Filter by date YYYY-MM-DD"),
    db: Session = Depends(get_db),
    auth_user: Optional[User] = Depends(get_current_user_or_ai_key),
):
    sessions = list_sessions(db, class_id=class_id, status=status, date=date)
    return [_format_session_response(s) for s in sessions]


@router.post(
    "",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Session",
    description="Schedule a new session for a class.",
)
def add_session(
    session_in: SessionCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("ADMIN", "TEACHER")),
):
    session_obj = create_session(db, session_in)
    return _format_session_response(session_obj)


@router.get(
    "/{id}",
    response_model=SessionResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Session Details",
    description="Retrieve a single session by its database ID.",
)
def get_session(
    id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session_obj = get_session_by_id(db, id)
    return _format_session_response(session_obj)


@router.post(
    "/{id}/start",
    response_model=SessionResponse,
    status_code=status.HTTP_200_OK,
    summary="Start Session",
    description="Activates a SCHEDULED session. Validates that no other ACTIVE session exists for this class.",
)
def activate_session(
    id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("ADMIN", "TEACHER")),
):
    session_obj = start_session(db, id)
    return _format_session_response(session_obj)


@router.post(
    "/{id}/close",
    response_model=SessionResponse,
    status_code=status.HTTP_200_OK,
    summary="Close Session",
    description="Completes an ACTIVE session and automatically generates ABSENT records for enrolled students.",
)
def complete_session(
    id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("ADMIN", "TEACHER")),
):
    session_obj = close_session(db, id)
    return _format_session_response(session_obj)


@router.post(
    "/{id}/cancel",
    response_model=SessionResponse,
    status_code=status.HTTP_200_OK,
    summary="Cancel Session",
    description="Cancels a SCHEDULED session.",
)
def cancel_session_route(
    id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("ADMIN", "TEACHER")),
):
    session_obj = cancel_session(db, id)
    return _format_session_response(session_obj)


@router.get(
    "/{id}/attendance",
    response_model=List[AttendanceRecordResponse],
    status_code=status.HTTP_200_OK,
    summary="Get Session Attendance",
    description="Retrieves the full attendance roster recorded for this session.",
)
def get_attendance_for_session(
    id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return get_session_attendance(db, id)
