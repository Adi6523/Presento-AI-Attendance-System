from __future__ import annotations

from .auth import LoginRequest, TokenResponse, UserBrief, TeacherBrief
from .user import UserCreate, UserResponse, TeacherCreate, TeacherResponse
from .student import StudentCreate, StudentUpdate, StudentResponse
from .class_schema import ClassCreate, ClassUpdate, ClassResponse, EnrollmentResponse
from .session import SessionCreate, SessionResponse
from .attendance import AttendanceEventIn, AttendanceEventOut, AttendanceRecordResponse
from .dashboard import DashboardSummary, RecentAttendanceResponse

__all__ = [
    "LoginRequest",
    "TokenResponse",
    "UserBrief",
    "TeacherBrief",
    "UserCreate",
    "UserResponse",
    "TeacherCreate",
    "TeacherResponse",
    "StudentCreate",
    "StudentUpdate",
    "StudentResponse",
    "ClassCreate",
    "ClassUpdate",
    "ClassResponse",
    "EnrollmentResponse",
    "SessionCreate",
    "SessionResponse",
    "AttendanceEventIn",
    "AttendanceEventOut",
    "AttendanceRecordResponse",
    "DashboardSummary",
    "RecentAttendanceResponse",
]
