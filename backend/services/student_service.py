from __future__ import annotations

from typing import List, Optional
from sqlalchemy import or_
from sqlalchemy.orm import Session

from ..exceptions import ConflictException, NotFoundException
from ..models import Attendance, Student
from ..schemas.student import StudentCreate, StudentUpdate


def get_student_by_id(db: Session, id: int) -> Student:
    student = db.query(Student).filter(Student.id == id).first()
    if not student:
        raise NotFoundException(f"Student with ID {id} not found", error_code="STUDENT_NOT_FOUND")
    return student


def get_student_by_student_id(db: Session, student_id: str) -> Optional[Student]:
    return db.query(Student).filter(Student.student_id == student_id).first()


def list_students(
    db: Session,
    student_id: Optional[str] = None,
    name: Optional[str] = None,
    active_only: bool = False,
) -> List[Student]:
    query = db.query(Student)

    if student_id:
        query = query.filter(Student.student_id.ilike(f"%{student_id}%"))
    if name:
        query = query.filter(Student.name.ilike(f"%{name}%"))
    if active_only:
        query = query.filter(Student.is_active.is_(True))

    return query.order_by(Student.id.asc()).all()


def create_student(db: Session, student_in: StudentCreate) -> Student:
    existing = get_student_by_student_id(db, student_in.student_id)
    if existing:
        raise ConflictException(
            f"Student with ID '{student_in.student_id}' already exists",
            error_code="STUDENT_ID_EXISTS",
        )

    student = Student(
        student_id=student_in.student_id,
        name=student_in.name,
        email=student_in.email,
        department=student_in.department,
        year=student_in.year,
        is_active=True,
    )
    db.add(student)
    db.commit()
    db.refresh(student)
    return student


def update_student(db: Session, id: int, student_in: StudentUpdate) -> Student:
    student = get_student_by_id(db, id)

    if student_in.name is not None:
        student.name = student_in.name
    if student_in.email is not None:
        student.email = student_in.email
    if student_in.department is not None:
        student.department = student_in.department
    if student_in.year is not None:
        student.year = student_in.year
    if student_in.is_active is not None:
        student.is_active = student_in.is_active

    db.commit()
    db.refresh(student)
    return student


def delete_student(db: Session, id: int) -> dict:
    student = get_student_by_id(db, id)

    # Check if attendance history exists
    has_attendance = db.query(Attendance).filter(Attendance.student_id == student.id).first() is not None

    if has_attendance:
        # Soft delete
        student.is_active = False
        db.commit()
        return {
            "success": True,
            "message": f"Student '{student.student_id}' deactivated (attendance history preserved)",
            "soft_deleted": True,
        }
    else:
        # Physical delete allowed
        db.delete(student)
        db.commit()
        return {
            "success": True,
            "message": f"Student '{student.student_id}' deleted",
            "soft_deleted": False,
        }
