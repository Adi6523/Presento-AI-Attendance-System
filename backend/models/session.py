from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, List, Optional
from sqlalchemy import DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..database import Base

if TYPE_CHECKING:
    from .class_model import ClassModel
    from .teacher import Teacher
    from .attendance import Attendance


class SessionModel(Base):
    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    class_id: Mapped[int] = mapped_column(
        ForeignKey("classes.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    teacher_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("teachers.id", ondelete="SET NULL"),
        nullable=True,
    )
    date: Mapped[str] = mapped_column(String(20), nullable=False)  # ISO Date YYYY-MM-DD
    start_time: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)  # HH:MM:SS
    end_time: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)  # HH:MM:SS
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="SCHEDULED",
        index=True,
    )  # SCHEDULED, ACTIVE, COMPLETED, CANCELLED
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    class_model: Mapped["ClassModel"] = relationship("ClassModel", back_populates="sessions")
    teacher: Mapped[Optional["Teacher"]] = relationship("Teacher", back_populates="sessions")
    attendances: Mapped[List["Attendance"]] = relationship(
        "Attendance",
        back_populates="session",
        cascade="all, delete-orphan",
    )
