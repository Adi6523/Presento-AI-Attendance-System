from __future__ import annotations

from typing import List
from pydantic import BaseModel
from .attendance import AttendanceRecordResponse


class DashboardSummary(BaseModel):
    total_students: int
    total_classes: int
    today_sessions: int
    present_today: int
    absent_today: int
    attendance_percentage: float


class RecentAttendanceResponse(BaseModel):
    records: List[AttendanceRecordResponse]
