from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, List, Optional
from sqlalchemy import DateTime, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..database import Base

if TYPE_CHECKING:
    from .teacher import Teacher
    from .student import Student
    from .session import SessionModel


class ClassModel(Base):
    __tablename__ = "classes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    subject: Mapped[str] = mapped_column(String(64), nullable=False)
    teacher_id: Mapped[Optional[int]] = mapped_column(ForeignKey("teachers.id", ondelete="SET NULL"), nullable=True)
    academic_year: Mapped[str] = mapped_column(String(32), nullable=False)
    semester: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    teacher: Mapped[Optional["Teacher"]] = relationship("Teacher", back_populates="classes")
    enrollments: Mapped[List["ClassStudent"]] = relationship(
        "ClassStudent",
        back_populates="class_model",
        cascade="all, delete-orphan",
    )
    sessions: Mapped[List["SessionModel"]] = relationship(
        "SessionModel",
        back_populates="class_model",
        cascade="all, delete-orphan",
    )


class ClassStudent(Base):
    __tablename__ = "class_students"
    __table_args__ = (
        UniqueConstraint("class_id", "student_id", name="uq_class_student"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    class_id: Mapped[int] = mapped_column(ForeignKey("classes.id", ondelete="CASCADE"), nullable=False)
    student_id: Mapped[int] = mapped_column(ForeignKey("students.id", ondelete="CASCADE"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    class_model: Mapped["ClassModel"] = relationship("ClassModel", back_populates="enrollments")
    student: Mapped["Student"] = relationship("Student", back_populates="enrollments")
