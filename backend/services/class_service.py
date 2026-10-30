from __future__ import annotations

from typing import List, Optional
from sqlalchemy.orm import Session

from ..exceptions import ConflictException, NotFoundException
from ..models import ClassModel, ClassStudent, Student, Teacher
from ..schemas.class_schema import ClassCreate, ClassUpdate


def get_class_by_id(db: Session, id: int) -> ClassModel:
    class_obj = db.query(ClassModel).filter(ClassModel.id == id).first()
    if not class_obj:
        raise NotFoundException(f"Class with ID {id} not found", error_code="CLASS_NOT_FOUND")
    return class_obj


def list_classes(db: Session) -> List[ClassModel]:
    return db.query(ClassModel).order_by(ClassModel.id.asc()).all()


def create_class(db: Session, class_in: ClassCreate) -> ClassModel:
    if class_in.teacher_id is not None:
        teacher = db.query(Teacher).filter(Teacher.id == class_in.teacher_id).first()
        if not teacher:
            raise NotFoundException(f"Teacher with ID {class_in.teacher_id} not found", error_code="TEACHER_NOT_FOUND")

    new_class = ClassModel(
        name=class_in.name,
        subject=class_in.subject,
        teacher_id=class_in.teacher_id,
        academic_year=class_in.academic_year,
        semester=class_in.semester,
    )
    db.add(new_class)
    db.commit()
    db.refresh(new_class)
    return new_class


def update_class(db: Session, id: int, class_in: ClassUpdate) -> ClassModel:
    class_obj = get_class_by_id(db, id)

    if class_in.name is not None:
        class_obj.name = class_in.name
    if class_in.subject is not None:
        class_obj.subject = class_in.subject
    if class_in.academic_year is not None:
        class_obj.academic_year = class_in.academic_year
    if class_in.semester is not None:
        class_obj.semester = class_in.semester
    if class_in.teacher_id is not None:
        teacher = db.query(Teacher).filter(Teacher.id == class_in.teacher_id).first()
        if not teacher:
            raise NotFoundException(f"Teacher with ID {class_in.teacher_id} not found", error_code="TEACHER_NOT_FOUND")
        class_obj.teacher_id = class_in.teacher_id

    db.commit()
    db.refresh(class_obj)
    return class_obj


def _find_student(db: Session, identifier: str) -> Student:
    """Finds student by integer primary key or student_id string."""
    student = None
    if identifier.isdigit():
        student = db.query(Student).filter(Student.id == int(identifier)).first()
    if not student:
        student = db.query(Student).filter(Student.student_id == identifier).first()
    if not student:
        raise NotFoundException(f"Student '{identifier}' not found", error_code="STUDENT_NOT_FOUND")
    return student


def enroll_student(db: Session, class_id: int, student_identifier: str) -> ClassStudent:
    class_obj = get_class_by_id(db, class_id)
    student = _find_student(db, student_identifier)

    # Check duplicate enrollment
    existing = (
        db.query(ClassStudent)
        .filter(ClassStudent.class_id == class_obj.id, ClassStudent.student_id == student.id)
        .first()
    )
    if existing:
        raise ConflictException(
            f"Student '{student.student_id}' is already enrolled in class '{class_obj.name}'",
            error_code="STUDENT_ALREADY_ENROLLED",
        )

    enrollment = ClassStudent(
        class_id=class_obj.id,
        student_id=student.id,
    )
    db.add(enrollment)
    db.commit()
    db.refresh(enrollment)
    return enrollment


def remove_student(db: Session, class_id: int, student_identifier: str) -> dict:
    class_obj = get_class_by_id(db, class_id)
    student = _find_student(db, student_identifier)

    enrollment = (
        db.query(ClassStudent)
        .filter(ClassStudent.class_id == class_obj.id, ClassStudent.student_id == student.id)
        .first()
    )
    if not enrollment:
        raise NotFoundException(
            f"Student '{student.student_id}' is not enrolled in class '{class_obj.name}'",
            error_code="STUDENT_NOT_ENROLLED",
        )

    db.delete(enrollment)
    db.commit()
    return {
        "success": True,
        "message": f"Student '{student.student_id}' removed from class '{class_obj.name}'",
    }
