from __future__ import annotations

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field

from .student import StudentResponse


class ClassCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=120)
    subject: str = Field(..., min_length=2, max_length=64)
    teacher_id: Optional[int] = None
    academic_year: str = Field(..., min_length=4, max_length=32)
    semester: int = Field(..., ge=1, le=12)


class ClassUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=2, max_length=120)
    subject: Optional[str] = Field(None, min_length=2, max_length=64)
    teacher_id: Optional[int] = None
    academic_year: Optional[str] = Field(None, min_length=4, max_length=32)
    semester: Optional[int] = Field(None, ge=1, le=12)


class ClassResponse(BaseModel):
    id: int
    name: str
    subject: str
    teacher_id: Optional[int] = None
    academic_year: str
    semester: int
    created_at: datetime
    enrolled_count: int = 0
    students: Optional[List[StudentResponse]] = None

    model_config = {"from_attributes": True}


class EnrollmentResponse(BaseModel):
    id: int
    class_id: int
    student_id: int
    created_at: datetime
    student: Optional[StudentResponse] = None

    model_config = {"from_attributes": True}
