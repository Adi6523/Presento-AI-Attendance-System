from __future__ import annotations

from typing import List
from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import ClassModel, User
from ..schemas.class_schema import ClassCreate, ClassResponse, ClassUpdate, EnrollmentResponse
from ..services.class_service import (
    create_class,
    enroll_student,
    get_class_by_id,
    list_classes,
    remove_student,
    update_class,
)
from .dependencies import get_current_user, require_role

router = APIRouter(prefix="/api/classes", tags=["Classes"])


def _format_class_response(cls: ClassModel) -> ClassResponse:
    return ClassResponse(
        id=cls.id,
        name=cls.name,
        subject=cls.subject,
        teacher_id=cls.teacher_id,
        academic_year=cls.academic_year,
        semester=cls.semester,
        created_at=cls.created_at,
        enrolled_count=len(cls.enrollments),
        students=[e.student for e in cls.enrollments],
    )


@router.get(
    "",
    response_model=List[ClassResponse],
    status_code=status.HTTP_200_OK,
    summary="List Classes",
    description="Retrieve all academic classes with enrollment counts.",
)
def get_classes(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    classes = list_classes(db)
    return [_format_class_response(c) for c in classes]


@router.post(
    "",
    response_model=ClassResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Class",
    description="Create a new class for a subject and semester.",
)
def add_class(
    class_in: ClassCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("ADMIN", "TEACHER")),
):
    cls = create_class(db, class_in)
    return _format_class_response(cls)


@router.get(
    "/{id}",
    response_model=ClassResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Class Details",
    description="Fetch single class along with enrolled student roster.",
)
def get_class(
    id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    cls = get_class_by_id(db, id)
    return _format_class_response(cls)


@router.put(
    "/{id}",
    response_model=ClassResponse,
    status_code=status.HTTP_200_OK,
    summary="Update Class",
    description="Update class information.",
)
def edit_class(
    id: int,
    class_in: ClassUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("ADMIN", "TEACHER")),
):
    cls = update_class(db, id, class_in)
    return _format_class_response(cls)


@router.post(
    "/{id}/students/{student_id}",
    response_model=EnrollmentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Enroll Student into Class",
    description="Enrolls a student into a class by numeric ID or student_id string. Prevents duplicates.",
)
def enroll_student_to_class(
    id: int,
    student_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("ADMIN", "TEACHER")),
):
    enrollment = enroll_student(db, class_id=id, student_identifier=student_id)
    return enrollment


@router.delete(
    "/{id}/students/{student_id}",
    status_code=status.HTTP_200_OK,
    summary="Remove Student from Class",
    description="Removes student enrollment from a class.",
)
def remove_student_from_class(
    id: int,
    student_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("ADMIN", "TEACHER")),
):
    return remove_student(db, class_id=id, student_identifier=student_id)
