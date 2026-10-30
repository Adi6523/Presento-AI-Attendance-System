"""
Database initialization utility.
Usage:
    python -m backend.init_db
"""
from __future__ import annotations

import logging
from sqlalchemy import inspect
from backend.database import engine, Base
from backend.models import (
    User,
    Teacher,
    Student,
    ClassModel,
    ClassStudent,
    SessionModel,
    Attendance,
    FaceProfile,
    FaceEmbedding,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("presento.init_db")

EXPECTED_TABLES = {
    "users",
    "teachers",
    "students",
    "classes",
    "class_students",
    "sessions",
    "attendance",
    "face_profiles",
    "face_embeddings",
}


def init_database() -> bool:
    """Creates tables if they do not exist and verifies schema integrity."""
    logger.info("Initializing database tables...")
    Base.metadata.create_all(bind=engine)

    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    missing_tables = EXPECTED_TABLES - existing_tables

    if missing_tables:
        logger.error("Schema verification failed! Missing tables: %s", missing_tables)
        return False

    logger.info("Schema verified successfully. All %d tables present: %s", len(EXPECTED_TABLES), sorted(list(existing_tables)))
    return True


if __name__ == "__main__":
    success = init_database()
    if not success:
        exit(1)
