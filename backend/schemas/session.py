from __future__ import annotations

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class SessionCreate(BaseModel):
    class_id: int
    teacher_id: Optional[int] = None
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    start_time: Optional[str] = Field(None, description="Start time in HH:MM:SS format")
    end_time: Optional[str] = Field(None, description="End time in HH:MM:SS format")


class SessionResponse(BaseModel):
    id: int
    class_id: int
    teacher_id: Optional[int] = None
    date: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    status: str
    created_at: datetime
    class_name: Optional[str] = None
    subject: Optional[str] = None

    model_config = {"from_attributes": True}
