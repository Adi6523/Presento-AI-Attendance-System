from __future__ import annotations

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=2, max_length=64)
    password: str = Field(..., min_length=4)


class TeacherBrief(BaseModel):
    id: int
    user_id: int
    name: str
    email: str

    model_config = {"from_attributes": True}


class UserBrief(BaseModel):
    id: int
    username: str
    role: str
    is_active: bool
    created_at: datetime
    teacher: Optional[TeacherBrief] = None

    model_config = {"from_attributes": True}


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserBrief
