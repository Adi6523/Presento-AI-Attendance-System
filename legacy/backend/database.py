from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

DATA_DIR = Path(os.getenv("PRESENTO_DATA_DIR", "data"))
DB_PATH = DATA_DIR / "presento.db"

SCHEMA = """
PRAGMA foreign_keys = ON;

-- ── Core tables (original) ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS students (
  id TEXT PRIMARY KEY, name TEXT NOT NULL, email TEXT, embedding TEXT NOT NULL,
  active INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  enrollment_quality REAL,           -- Average sample quality (0-1)
  enrollment_samples INTEGER,        -- Number of valid samples used
  enrolled_at TEXT,                  -- ISO timestamp of enrollment
  embedding_version INTEGER DEFAULT 1 -- For future model changes
);
CREATE TABLE IF NOT EXISTS sessions (
  id INTEGER PRIMARY KEY AUTOINCREMENT, course_code TEXT NOT NULL, room TEXT NOT NULL,
  starts_at TEXT NOT NULL, ends_at TEXT NOT NULL, late_after_minutes INTEGER NOT NULL DEFAULT 10
);
CREATE TABLE IF NOT EXISTS attendance (
  id INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT NOT NULL UNIQUE, student_id TEXT NOT NULL,
  session_id INTEGER, device_id TEXT NOT NULL, observed_at TEXT NOT NULL, status TEXT NOT NULL,
  confidence REAL NOT NULL, liveness_score REAL NOT NULL,
  reason TEXT, manually_added INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY(student_id) REFERENCES students(id),
  FOREIGN KEY(session_id) REFERENCES sessions(id)
);
CREATE INDEX IF NOT EXISTS idx_attendance_student_day ON attendance(student_id, observed_at);
CREATE TABLE IF NOT EXISTS users (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  username TEXT NOT NULL UNIQUE,
  email TEXT UNIQUE,
  password_hash TEXT NOT NULL,
  role TEXT NOT NULL DEFAULT 'student',
  student_id TEXT,
  active INTEGER NOT NULL DEFAULT 1,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY(student_id) REFERENCES students(id)
);

-- ── Academic structure ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS departments (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  code TEXT NOT NULL UNIQUE,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS academic_years (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  label TEXT NOT NULL UNIQUE,
  is_current INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS classes (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  department_id INTEGER NOT NULL,
  academic_year_id INTEGER NOT NULL,
  year TEXT NOT NULL,
  semester INTEGER,
  division TEXT NOT NULL,
  label TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY(department_id) REFERENCES departments(id),
  FOREIGN KEY(academic_year_id) REFERENCES academic_years(id)
);
CREATE TABLE IF NOT EXISTS subjects (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  department_id INTEGER NOT NULL,
  name TEXT NOT NULL,
  code TEXT NOT NULL UNIQUE,
  semester INTEGER,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY(department_id) REFERENCES departments(id)
);
CREATE TABLE IF NOT EXISTS teacher_profiles (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER NOT NULL UNIQUE,
  department_id INTEGER,
  employee_id TEXT,
  designation TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY(user_id) REFERENCES users(id),
  FOREIGN KEY(department_id) REFERENCES departments(id)
);
CREATE TABLE IF NOT EXISTS student_profiles (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  student_id TEXT NOT NULL,
  class_id INTEGER NOT NULL,
  roll_number TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY(student_id) REFERENCES students(id),
  FOREIGN KEY(class_id) REFERENCES classes(id),
  UNIQUE(student_id, class_id)
);
CREATE TABLE IF NOT EXISTS teacher_assignments (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  teacher_id INTEGER NOT NULL,
  subject_id INTEGER NOT NULL,
  class_id INTEGER NOT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY(teacher_id) REFERENCES teacher_profiles(id),
  FOREIGN KEY(subject_id) REFERENCES subjects(id),
  FOREIGN KEY(class_id) REFERENCES classes(id),
  UNIQUE(teacher_id, subject_id, class_id)
);

-- ── Audit log ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS audit_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  actor_id INTEGER NOT NULL,
  actor_username TEXT NOT NULL,
  action TEXT NOT NULL,
  target_type TEXT,
  target_id TEXT,
  details TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""

# Safe migrations applied on every startup — silently skipped if columns exist
_MIGRATIONS = [
    # Original migrations
    "ALTER TABLE attendance ADD COLUMN reason TEXT;",
    "ALTER TABLE attendance ADD COLUMN manually_added INTEGER NOT NULL DEFAULT 0;",
    # Phase 1: Enhanced enrollment metadata on students
    "ALTER TABLE students ADD COLUMN enrollment_quality REAL;",
    "ALTER TABLE students ADD COLUMN enrollment_samples INTEGER;",
    "ALTER TABLE students ADD COLUMN enrolled_at TEXT;",
    "ALTER TABLE students ADD COLUMN embedding_version INTEGER DEFAULT 1;",
    # Phase 1: Last-login tracking on users
    "ALTER TABLE users ADD COLUMN last_login TEXT;",
    # Phase 3: Academic context on sessions (nullable for backward compat)
    "ALTER TABLE sessions ADD COLUMN teacher_id INTEGER REFERENCES teacher_profiles(id);",
    "ALTER TABLE sessions ADD COLUMN subject_id INTEGER REFERENCES subjects(id);",
    "ALTER TABLE sessions ADD COLUMN class_id INTEGER REFERENCES classes(id);",
    "ALTER TABLE sessions ADD COLUMN status TEXT NOT NULL DEFAULT 'open';",
    # Phase 4: Attendance modification tracking
    "ALTER TABLE attendance ADD COLUMN modified_by TEXT;",
    "ALTER TABLE attendance ADD COLUMN modified_at TEXT;",
    "ALTER TABLE attendance ADD COLUMN original_status TEXT;",
    # Phase 4: Enhanced attendance tracking
    "ALTER TABLE attendance ADD COLUMN recognition_processing_time INTEGER;",
    "ALTER TABLE attendance ADD COLUMN liveness_confidence REAL;",
    "ALTER TABLE attendance ADD COLUMN face_quality_score REAL;",
]


def initialize() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with connect() as db:
        db.executescript(SCHEMA)
        for sql in _MIGRATIONS:
            try:
                db.execute(sql)
            except Exception:
                pass  # column already exists — safe to ignore


@contextmanager
def connect():
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    try:
        yield db
        db.commit()
    finally:
        db.close()


def log_audit(
    db: sqlite3.Connection,
    actor_id: int,
    actor_username: str,
    action: str,
    target_type: str | None = None,
    target_id: str | None = None,
    details: str | None = None,
) -> None:
    db.execute(
        "INSERT INTO audit_log(actor_id, actor_username, action, target_type, target_id, details) "
        "VALUES(?, ?, ?, ?, ?, ?)",
        (actor_id, actor_username, action, target_type, str(target_id) if target_id is not None else None, details)
    )

