from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy import exc
from sqlalchemy.orm import Session

from ..config import settings
from ..exceptions import NotFoundException
from ..models import Attendance, ClassStudent, SessionModel, Student
from ..schemas.attendance import AttendanceEventIn, AttendanceEventOut, AttendanceRecordResponse

logger = logging.getLogger("presento.attendance_service")


def parse_event_timestamp(ts: Optional[datetime | str]) -> datetime:
    """Parses incoming ISO timestamp string or returns current UTC datetime."""
    if ts is None:
        return datetime.utcnow()
    if isinstance(ts, datetime):
        return ts
    try:
        # Handle ISO strings
        clean_str = ts.replace("Z", "+00:00")
        dt = datetime.fromisoformat(clean_str)
        # Convert to naive UTC for consistent DB storage
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except Exception:
        return datetime.utcnow()


def process_attendance_event(db: Session, event_in: AttendanceEventIn) -> AttendanceEventOut:
    """
    9-step validation pipeline for incoming AI attendance events.
    Enforces strict business rules before admitting PRESENT attendance.
    """
    # ── STEP 1: Check student exists ──────────────────────────────────────────
    if not event_in.student_id:
        logger.info("Event rejected: Missing student_id")
        return AttendanceEventOut(
            success=False,
            status="BLOCKED",
            message="No student identity provided (unknown face)",
        )

    student = db.query(Student).filter(Student.student_id == event_in.student_id).first()
    if not student:
        logger.info("Event rejected: Student '%s' does not exist", event_in.student_id)
        return AttendanceEventOut(
            success=False,
            status="REJECTED",
            message=f"Student ID '{event_in.student_id}' not found in database",
        )

    # ── STEP 2: Check student is active ───────────────────────────────────────
    if not student.is_active:
        logger.info("Event rejected: Student '%s' is inactive", student.student_id)
        return AttendanceEventOut(
            success=False,
            status="REJECTED",
            message=f"Student '{student.student_id}' is marked inactive",
            student_id=student.student_id,
            student_name=student.name,
        )

    # ── STEP 3: Check session exists ──────────────────────────────────────────
    session_obj = db.query(SessionModel).filter(SessionModel.id == event_in.session_id).first()
    if not session_obj:
        logger.info("Event rejected: Session ID %d does not exist", event_in.session_id)
        return AttendanceEventOut(
            success=False,
            status="REJECTED",
            message=f"Session with ID {event_in.session_id} not found",
            student_id=student.student_id,
            student_name=student.name,
        )

    # ── STEP 4: Check session status is ACTIVE ────────────────────────────────
    if session_obj.status != "ACTIVE":
        logger.info("Event rejected: Session ID %d status is '%s', not ACTIVE", session_obj.id, session_obj.status)
        return AttendanceEventOut(
            success=False,
            status="REJECTED",
            message=f"Session is in state '{session_obj.status}'. Only ACTIVE sessions accept attendance.",
            student_id=student.student_id,
            student_name=student.name,
        )

    # ── STEP 5: Check student is enrolled in the session's class ──────────────
    enrollment = (
        db.query(ClassStudent)
        .filter(
            ClassStudent.class_id == session_obj.class_id,
            ClassStudent.student_id == student.id,
        )
        .first()
    )
    if not enrollment:
        logger.info("Event rejected: Student '%s' not enrolled in class ID %d", student.student_id, session_obj.class_id)
        return AttendanceEventOut(
            success=False,
            status="REJECTED",
            message=f"Student '{student.student_id}' is not enrolled in this session's class",
            student_id=student.student_id,
            student_name=student.name,
        )

    # ── STEP 6: Check confidence threshold ────────────────────────────────────
    min_conf = settings.attendance_min_confidence
    if event_in.confidence < min_conf:
        logger.info("Event blocked: Confidence %.3f below threshold %.3f", event_in.confidence, min_conf)
        return AttendanceEventOut(
            success=False,
            status="BLOCKED",
            message=f"Confidence ({event_in.confidence:.2f}) below operating threshold ({min_conf:.2f})",
            student_id=student.student_id,
            student_name=student.name,
        )

    # ── STEP 7: Validate liveness via AI temporal state ──────────────────────
    liveness_state = event_in.liveness_state  # Already uppercased by validator
    if liveness_state != "REAL_LOCKED":
        if liveness_state == "SPOOF":
            reason = "Presentation attack detected (SPOOF state). Attendance blocked."
        elif liveness_state in {"CHECKING", "HOLD"}:
            reason = f"Liveness verification incomplete (state: {liveness_state}). Attendance blocked."
        else:
            reason = f"Liveness state '{liveness_state}' is not verified real. Only REAL_LOCKED is accepted."
        logger.info(
            "Event blocked: liveness_state='%s' for student '%s'. Reason: %s",
            liveness_state, student.student_id, reason,
        )
        return AttendanceEventOut(
            success=False,
            status="BLOCKED",
            message=reason,
            student_id=student.student_id,
            student_name=student.name,
        )

    # ── STEP 8: Check duplicate (application level) ───────────────────────────
    existing = (
        db.query(Attendance)
        .filter(
            Attendance.session_id == session_obj.id,
            Attendance.student_id == student.id,
        )
        .first()
    )
    if existing:
        logger.info("Duplicate event: Student '%s' already recorded in session %d", student.student_id, session_obj.id)
        return AttendanceEventOut(
            success=True,
            status="DUPLICATE",
            message=f"Student '{student.student_id}' attendance already recorded for this session",
            attendance_id=existing.id,
            student_id=student.student_id,
            student_name=student.name,
        )

    # ── STEP 9: Insert PRESENT with race condition protection ──────────────────
    parsed_time = parse_event_timestamp(event_in.timestamp)
    attendance_record = Attendance(
        session_id=session_obj.id,
        student_id=student.id,
        timestamp=parsed_time,
        status="PRESENT",
        confidence=round(float(event_in.confidence), 4),
        liveness_score=round(float(event_in.liveness_score), 4) if event_in.liveness_score is not None else None,
        source=event_in.source or "AI_CAMERA",
    )
    db.add(attendance_record)

    try:
        db.commit()
        db.refresh(attendance_record)
        logger.info("Attendance RECORDED: Student '%s' PRESENT in session %d", student.student_id, session_obj.id)
        return AttendanceEventOut(
            success=True,
            status="PRESENT",
            message="Attendance recorded successfully",
            attendance_id=attendance_record.id,
            student_id=student.student_id,
            student_name=student.name,
        )
    except exc.IntegrityError:
        db.rollback()
        # Concurrent duplicate caught by UNIQUE(session_id, student_id)
        existing_concurrent = (
            db.query(Attendance)
            .filter(
                Attendance.session_id == session_obj.id,
                Attendance.student_id == student.id,
            )
            .first()
        )
        att_id = existing_concurrent.id if existing_concurrent else None
        return AttendanceEventOut(
            success=True,
            status="DUPLICATE",
            message=f"Student '{student.student_id}' attendance already recorded concurrently",
            attendance_id=att_id,
            student_id=student.student_id,
            student_name=student.name,
        )


def get_session_attendance(db: Session, session_id: int) -> List[AttendanceRecordResponse]:
    session_obj = db.query(SessionModel).filter(SessionModel.id == session_id).first()
    if not session_obj:
        raise NotFoundException(f"Session with ID {session_id} not found", error_code="SESSION_NOT_FOUND")

    records = (
        db.query(Attendance)
        .join(Student, Student.id == Attendance.student_id)
        .filter(Attendance.session_id == session_id)
        .order_by(Attendance.timestamp.asc())
        .all()
    )

    result = []
    for r in records:
        result.append(
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
        )
    return result


def get_attendance_history(
    db: Session,
    date: Optional[str] = None,
    class_id: Optional[int] = None,
    student_id: Optional[str] = None,
    status: Optional[str] = None,
) -> List[AttendanceRecordResponse]:
    query = (
        db.query(Attendance)
        .join(Student, Student.id == Attendance.student_id)
        .join(SessionModel, SessionModel.id == Attendance.session_id)
    )

    if class_id is not None:
        query = query.filter(SessionModel.class_id == class_id)
    if date is not None:
        query = query.filter(SessionModel.date == date)
    if student_id is not None:
        query = query.filter(Student.student_id.ilike(f"%{student_id}%"))
    if status is not None:
        query = query.filter(Attendance.status == status.upper())

    records = query.order_by(Attendance.timestamp.desc()).all()

    result = []
    for r in records:
        result.append(
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
        )
    return result
