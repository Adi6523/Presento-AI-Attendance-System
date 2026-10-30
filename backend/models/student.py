from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, List, Optional
from sqlalchemy import Boolean, DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..database import Base

if TYPE_CHECKING:
    from .class_model import ClassStudent
    from .attendance import Attendance
    from .face import FaceProfile


class Student(Base):
    __tablename__ = "students"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    student_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    email: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)
    department: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    year: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    enrollments: Mapped[List["ClassStudent"]] = relationship(
        "ClassStudent",
        back_populates="student",
        cascade="all, delete-orphan",
    )
    attendances: Mapped[List["Attendance"]] = relationship(
        "Attendance",
        back_populates="student",
        cascade="all, delete-orphan",
    )
    face_profile: Mapped[Optional["FaceProfile"]] = relationship(
        "FaceProfile",
        back_populates="student",
        uselist=False,
        cascade="all, delete-orphan",
    )
