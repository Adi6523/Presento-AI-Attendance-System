from __future__ import annotations

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class UserCreate(BaseModel):
    username: str = Field(..., min_length=2, max_length=64)
    password: str = Field(..., min_length=4)
    role: str = Field(default="TEACHER")  # ADMIN, TEACHER


class UserResponse(BaseModel):
    id: int
    username: str
    role: str
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class TeacherCreate(BaseModel):
    user_id: int
    name: str = Field(..., min_length=2, max_length=120)
    email: str = Field(..., min_length=3, max_length=120)


class TeacherResponse(BaseModel):
    id: int
    user_id: int
    name: str
    email: str
    created_at: datetime

    model_config = {"from_attributes": True}
