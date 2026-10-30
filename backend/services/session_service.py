from __future__ import annotations

from datetime import datetime
from typing import List, Optional
from sqlalchemy.orm import Session

from ..exceptions import BadRequestException, ConflictException, NotFoundException
from ..models import Attendance, ClassModel, ClassStudent, SessionModel, Student
from ..schemas.session import SessionCreate


def get_session_by_id(db: Session, id: int) -> SessionModel:
    session_obj = db.query(SessionModel).filter(SessionModel.id == id).first()
    if not session_obj:
        raise NotFoundException(f"Session with ID {id} not found", error_code="SESSION_NOT_FOUND")
    return session_obj


def list_sessions(
    db: Session,
    class_id: Optional[int] = None,
    status: Optional[str] = None,
    date: Optional[str] = None,
) -> List[SessionModel]:
    query = db.query(SessionModel)

    if class_id is not None:
        query = query.filter(SessionModel.class_id == class_id)
    if status is not None:
        query = query.filter(SessionModel.status == status.upper())
    if date is not None:
        query = query.filter(SessionModel.date == date)

    return query.order_by(SessionModel.id.desc()).all()


def create_session(db: Session, session_in: SessionCreate) -> SessionModel:
    class_obj = db.query(ClassModel).filter(ClassModel.id == session_in.class_id).first()
    if not class_obj:
        raise NotFoundException(f"Class with ID {session_in.class_id} not found", error_code="CLASS_NOT_FOUND")

    teacher_id = session_in.teacher_id or class_obj.teacher_id

    new_session = SessionModel(
        class_id=session_in.class_id,
        teacher_id=teacher_id,
        date=session_in.date,
        start_time=session_in.start_time,
        end_time=session_in.end_time,
        status="SCHEDULED",
    )
    db.add(new_session)
    db.commit()
    db.refresh(new_session)
    return new_session


def start_session(db: Session, id: int) -> SessionModel:
    """Transitions a session from SCHEDULED to ACTIVE. Enforces single active session per class."""
    session_obj = get_session_by_id(db, id)

    if session_obj.status != "SCHEDULED":
        raise BadRequestException(
            f"Cannot start session in '{session_obj.status}' state. Only SCHEDULED sessions can be started.",
            error_code="INVALID_STATE_TRANSITION",
        )

    # Verify no other ACTIVE session exists for this class
    active_exists = (
        db.query(SessionModel)
        .filter(
            SessionModel.class_id == session_obj.class_id,
            SessionModel.status == "ACTIVE",
            SessionModel.id != session_obj.id,
        )
        .first()
    )
    if active_exists:
        raise ConflictException(
            f"Class already has an ACTIVE session (Session ID {active_exists.id}). Finish it first.",
            error_code="ACTIVE_SESSION_EXISTS",
        )

    session_obj.status = "ACTIVE"
    if not session_obj.start_time:
        session_obj.start_time = datetime.utcnow().strftime("%H:%M:%S")

    db.commit()
    db.refresh(session_obj)
    return session_obj


def close_session(db: Session, id: int) -> SessionModel:
    """
    Transitions session from ACTIVE to COMPLETED.
    Generates ABSENT attendance records for enrolled active students without PRESENT record.
    """
    session_obj = get_session_by_id(db, id)

    if session_obj.status != "ACTIVE":
        raise BadRequestException(
            f"Cannot close session in '{session_obj.status}' state. Only ACTIVE sessions can be closed.",
            error_code="INVALID_STATE_TRANSITION",
        )

    # 1. Fetch all active students enrolled in this class
    enrolled_students = (
        db.query(Student)
        .join(ClassStudent, ClassStudent.student_id == Student.id)
        .filter(
            ClassStudent.class_id == session_obj.class_id,
            Student.is_active.is_(True),
        )
        .all()
    )

    # 2. Fetch existing attendance records for this session
    existing_attendance = (
        db.query(Attendance)
        .filter(Attendance.session_id == session_obj.id)
        .all()
    )
    attendance_by_student_id = {att.student_id: att for att in existing_attendance}

    now = datetime.utcnow()
    # 3. For every enrolled student, if no record exists, insert ABSENT
    for student in enrolled_students:
        if student.id not in attendance_by_student_id:
            absent_record = Attendance(
                session_id=session_obj.id,
                student_id=student.id,
                timestamp=now,
                status="ABSENT",
                confidence=None,
                liveness_score=None,
                source="MANUAL",
            )
            db.add(absent_record)

    session_obj.status = "COMPLETED"
    if not session_obj.end_time:
        session_obj.end_time = now.strftime("%H:%M:%S")

    db.commit()
    db.refresh(session_obj)
    return session_obj


def cancel_session(db: Session, id: int) -> SessionModel:
    """Transitions session from SCHEDULED to CANCELLED."""
    session_obj = get_session_by_id(db, id)

    if session_obj.status != "SCHEDULED":
        raise BadRequestException(
            f"Cannot cancel session in '{session_obj.status}' state. Only SCHEDULED sessions can be cancelled.",
            error_code="INVALID_STATE_TRANSITION",
        )

    session_obj.status = "CANCELLED"
    db.commit()
    db.refresh(session_obj)
    return session_obj
