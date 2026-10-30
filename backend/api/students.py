from __future__ import annotations

from typing import List, Optional
from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import User
from ..schemas.student import StudentCreate, StudentResponse, StudentUpdate
from ..services.student_service import (
    create_student,
    delete_student,
    get_student_by_id,
    list_students,
    update_student,
)
from .dependencies import get_current_user, require_role

router = APIRouter(prefix="/api/students", tags=["Students"])


@router.get(
    "",
    response_model=List[StudentResponse],
    status_code=status.HTTP_200_OK,
    summary="List and Search Students",
    description="Retrieve students with optional filtering by student_id or name.",
)
def get_students(
    student_id: Optional[str] = Query(None, description="Search filter for roll number / student_id"),
    name: Optional[str] = Query(None, description="Search filter for student name"),
    active_only: bool = Query(False, description="Filter only active students"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return list_students(db, student_id=student_id, name=name, active_only=active_only)


@router.post(
    "",
    response_model=StudentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Student",
    description="Register a new student with a unique student_id.",
)
def add_student(
    student_in: StudentCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("ADMIN", "TEACHER")),
):
    return create_student(db, student_in)


@router.get(
    "/{id}",
    response_model=StudentResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Student by ID",
    description="Fetch single student record by integer database ID.",
)
def get_student(
    id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return get_student_by_id(db, id)


@router.put(
    "/{id}",
    response_model=StudentResponse,
    status_code=status.HTTP_200_OK,
    summary="Update Student",
    description="Modify student profile details.",
)
def edit_student(
    id: int,
    student_in: StudentUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("ADMIN", "TEACHER")),
):
    return update_student(db, id, student_in)


@router.delete(
    "/{id}",
    status_code=status.HTTP_200_OK,
    summary="Delete Student",
    description="Deletes student or marks inactive (soft-delete) if attendance records exist.",
)
def remove_student(
    id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("ADMIN", "TEACHER")),
):
    return delete_student(db, id)
