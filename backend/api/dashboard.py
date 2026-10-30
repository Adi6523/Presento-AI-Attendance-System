from __future__ import annotations

from datetime import datetime
from typing import List
from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Attendance, ClassModel, SessionModel, Student, User
from ..schemas.attendance import AttendanceRecordResponse
from ..schemas.dashboard import DashboardSummary, RecentAttendanceResponse
from .dependencies import get_current_user

router = APIRouter(prefix="/api/dashboard", tags=["Dashboard"])


@router.get(
    "/summary",
    response_model=DashboardSummary,
    status_code=status.HTTP_200_OK,
    summary="Dashboard Summary",
    description="Returns aggregate statistics: total students, total classes, today's sessions, present, absent, and attendance percentage.",
)
def get_dashboard_summary(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    today_str = datetime.utcnow().strftime("%Y-%m-%d")

    total_students = db.query(Student).filter(Student.is_active.is_(True)).count()
    total_classes = db.query(ClassModel).count()
    today_sessions = db.query(SessionModel).filter(SessionModel.date == today_str).count()

    today_session_ids = [
        s.id for s in db.query(SessionModel.id).filter(SessionModel.date == today_str).all()
    ]

    if today_session_ids:
        present_today = (
            db.query(Attendance)
            .filter(Attendance.session_id.in_(today_session_ids), Attendance.status == "PRESENT")
            .count()
        )
        absent_today = (
            db.query(Attendance)
            .filter(Attendance.session_id.in_(today_session_ids), Attendance.status == "ABSENT")
            .count()
        )
    else:
        present_today = 0
        absent_today = 0

    total_marked = present_today + absent_today
    attendance_pct = round((present_today / total_marked * 100.0), 1) if total_marked > 0 else 0.0

    return DashboardSummary(
        total_students=total_students,
        total_classes=total_classes,
        today_sessions=today_sessions,
        present_today=present_today,
        absent_today=absent_today,
        attendance_percentage=attendance_pct,
    )


@router.get(
    "/recent-attendance",
    response_model=RecentAttendanceResponse,
    status_code=status.HTTP_200_OK,
    summary="Recent Attendance Records",
    description="Retrieves the most recent 15 attendance events across all sessions.",
)
def get_recent_attendance(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    recent_records = (
        db.query(Attendance)
        .join(Student, Student.id == Attendance.student_id)
        .order_by(Attendance.timestamp.desc())
        .limit(15)
        .all()
    )

    records = [
        AttendanceRecordResponse(
            id=r.id,
            session_id=r.session_id,
            student_id=r.student.student_id,
            student_name=r.student.name,
            timestamp=r.timestamp,
            status=r.status,
            confidence=r.confidence,
            liveness_score=r.liveness_score,
            source=r.source,
        )
        for r in recent_records
    ]

    return RecentAttendanceResponse(records=records)
