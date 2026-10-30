from __future__ import annotations

from .auth import router as auth_router
from .students import router as students_router
from .classes import router as classes_router
from .sessions import router as sessions_router
from .attendance import router as attendance_router
from .dashboard import router as dashboard_router

__all__ = [
    "auth_router",
    "students_router",
    "classes_router",
    "sessions_router",
    "attendance_router",
    "dashboard_router",
]
