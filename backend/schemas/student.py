from __future__ import annotations

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class StudentCreate(BaseModel):
    student_id: str = Field(..., min_length=2, max_length=64, description="Unique roll number or ID, e.g., 23UAM090")
    name: str = Field(..., min_length=2, max_length=120)
    email: Optional[str] = None
    department: Optional[str] = None
    year: Optional[str] = None


class StudentUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=2, max_length=120)
    email: Optional[str] = None
    department: Optional[str] = None
    year: Optional[str] = None
    is_active: Optional[bool] = None


class StudentResponse(BaseModel):
    id: int
    student_id: str
    name: str
    email: Optional[str] = None
    department: Optional[str] = None
    year: Optional[str] = None
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}
