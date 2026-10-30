from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, List
from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..database import Base

if TYPE_CHECKING:
    from .student import Student


class FaceProfile(Base):
    __tablename__ = "face_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    student_id: Mapped[int] = mapped_column(
        ForeignKey("students.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    student: Mapped["Student"] = relationship("Student", back_populates="face_profile")
    embeddings: Mapped[List["FaceEmbedding"]] = relationship(
        "FaceEmbedding",
        back_populates="face_profile",
        cascade="all, delete-orphan",
    )


class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    face_profile_id: Mapped[int] = mapped_column(
        ForeignKey("face_profiles.id", ondelete="CASCADE"),
        nullable=False,
    )
    embedding: Mapped[str] = mapped_column(Text, nullable=False)  # JSON-encoded float array
    embedding_dimension: Mapped[int] = mapped_column(Integer, nullable=False, default=512)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    face_profile: Mapped["FaceProfile"] = relationship("FaceProfile", back_populates="embeddings")
