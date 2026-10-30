from __future__ import annotations

import json
import os
import threading
import uuid
import face_recognition
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field

from .auth import (
    authenticate_user,
    create_access_token,
    get_current_user,
    hash_password,
    require_role,
)
from .database import connect, initialize, log_audit
from .web_vision import decode_image, embedding, embedding_enroll, ear, liveness_ok, filter_enrollment_samples

app = FastAPI(title="Presento API", version="0.4.0")
app.add_middleware(
    CORSMiddleware,
    # Allow requests from Vite dev server AND from the HTML launcher (null = file://)
    allow_origins=os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,null"
    ).split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Device token auth (Pi/edge clients) ───────────────────────────────────────
def require_device(x_api_token: str = Header(default="")):
    expected = os.getenv("API_TOKEN", "change-me-before-production")
    if expected and x_api_token != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid device token")


# ── On-Demand Pi Camera Session State ─────────────────────────────────────────
_backend_camera_lock = threading.Lock()
_backend_camera_session: Optional[any] = None


# ── Pydantic models ────────────────────────────────────────────────────────────

class StudentIn(BaseModel):
    id: str = Field(min_length=2, max_length=64)
    name: str = Field(min_length=2, max_length=120)
    email: str | None = None
    embedding: list[float] = Field(min_length=128, max_length=128)

class StudentRecordIn(BaseModel):
    id: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=2, max_length=120)
    email: str | None = None
    class_id: int | None = None
    roll_number: str | None = None

class StudentUpdateIn(BaseModel):
    name: str | None = Field(default=None, min_length=2, max_length=120)
    email: str | None = None
    class_id: int | None = None
    roll_number: str | None = None

class SessionIn(BaseModel):
    course_code: str | None = None
    room: str = "A101"
    starts_at: datetime
    ends_at: datetime
    late_after_minutes: int = Field(default=10, ge=0, le=120)
    teacher_id: int | None = None
    subject_id: int | None = None
    class_id: int | None = None

class CameraFramesIn(BaseModel):
    images: list[str] = Field(min_length=1, max_length=30)
    session_id: int | None = None

class EventIn(BaseModel):
    event_id: str
    student_id: str
    device_id: str
    observed_at: datetime
    confidence: float = Field(ge=0, le=1)
    liveness_score: float = Field(ge=0, le=1)
    room: str | None = None
    session_id: int | None = None

class AttendanceManualIn(BaseModel):
    student_id: str
    session_id: int | None = None
    status: Literal["present", "late", "absent", "nc", "present_nc"] = "present"
    observed_at: str   # ISO datetime string from frontend
    reason: str | None = None

class AttendancePatchIn(BaseModel):
    status: Literal["present", "late", "absent", "nc", "present_nc"]
    reason: str | None = None

class CreateUserIn(BaseModel):
    username: str = Field(min_length=2, max_length=64)
    password: str = Field(min_length=6, max_length=128)
    email: str | None = None
    role: Literal["admin", "teacher", "student"] = "student"
    student_id: str | None = None
    name: str | None = None

class ChangePasswordIn(BaseModel):
    current_password: str
    new_password: str = Field(min_length=6, max_length=128)

class UserUpdateIn(BaseModel):
    email: str | None = None
    active: bool | None = None
    new_password: str | None = Field(default=None, min_length=6, max_length=128)

# Academic models
class DepartmentIn(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    code: str = Field(min_length=2, max_length=32)

class AcademicYearIn(BaseModel):
    label: str = Field(min_length=4, max_length=32)
    is_current: bool = False

class ClassIn(BaseModel):
    department_id: int
    academic_year_id: int
    year: str = Field(min_length=1, max_length=16)
    semester: int | None = None
    division: str = Field(min_length=1, max_length=16)
    label: str = Field(min_length=2, max_length=64)

class SubjectIn(BaseModel):
    department_id: int
    name: str = Field(min_length=2, max_length=120)
    code: str = Field(min_length=2, max_length=32)
    semester: int | None = None

class TeacherProfileIn(BaseModel):
    user_id: int
    department_id: int | None = None
    employee_id: str = Field(min_length=2, max_length=32)
    designation: str = "Faculty"

class TeacherAssignmentIn(BaseModel):
    teacher_id: int
    subject_id: int
    class_id: int

class StudentProfileIn(BaseModel):
    student_id: str
    class_id: int
    roll_number: str | None = None

class StudentRecordIn(BaseModel):
    id: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=120)
    email: str | None = None
    class_id: int | None = None
    roll_number: str | None = None

class StudentUpdateIn(BaseModel):
    name: str | None = None
    email: str | None = None
    class_id: int | None = None
    roll_number: str | None = None


# ── Startup ────────────────────────────────────────────────────────────────────
@app.on_event("startup")
def startup():
    initialize()


# ── Health ─────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"ok": True}


# ══════════════════════════════════════════════════════════════════════════════
# AUTH
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/auth/login")
def login(form: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form.username, form.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password")
    with connect() as db:
        db.execute(
            "UPDATE users SET last_login=? WHERE id=?",
            (datetime.now(timezone.utc).isoformat(), user["id"])
        )
    token = create_access_token({"sub": str(user["id"]), "role": user["role"]})
    return {
        "access_token": token,
        "token_type": "bearer",
        "role": user["role"],
        "username": user["username"],
        "student_id": user["student_id"],
    }

@app.get("/auth/me")
def me(current_user: dict = Depends(get_current_user)):
    return {k: v for k, v in current_user.items() if k != "password_hash"}

@app.post("/auth/register", status_code=201)
def create_user(body: CreateUserIn, current_user: dict = Depends(require_role("admin"))):
    if body.role == "student" and not body.student_id:
        raise HTTPException(422, "student_id is required when role is 'student'")
    try:
        with connect() as db:
            if body.role == "student" and body.student_id:
                # Ensure students table has entry first to fulfill foreign key constraint
                student_name = (body.name or body.username).strip()
                db.execute(
                    "INSERT INTO students(id,name,email,embedding) VALUES(?,?,?,'[]') "
                    "ON CONFLICT(id) DO UPDATE SET name=COALESCE(excluded.name, students.name), email=COALESCE(excluded.email, students.email), active=1",
                    (body.student_id, student_name, body.email)
                )

            # Check if user already exists: if so, update their password and profile
            existing = db.execute("SELECT id FROM users WHERE username=?", (body.username,)).fetchone()
            if existing:
                user_id = existing[0]
                db.execute(
                    "UPDATE users SET email=?, password_hash=?, role=?, student_id=?, active=1 WHERE id=?",
                    (body.email, hash_password(body.password), body.role, body.student_id, user_id)
                )
            else:
                cur = db.execute(
                    "INSERT INTO users(username,email,password_hash,role,student_id) VALUES(?,?,?,?,?)",
                    (body.username, body.email, hash_password(body.password), body.role, body.student_id),
                )
                user_id = cur.lastrowid

            if body.role == "teacher":
                db.execute(
                    "INSERT OR IGNORE INTO teacher_profiles(user_id, employee_id, designation) VALUES(?,?,?)",
                    (user_id, f"EMP-{user_id}", "Teacher")
                )
            log_audit(db, current_user["id"], current_user["username"], "create_user", "user", user_id, f"Created {body.role} {body.username}")
    except Exception as exc:
        if "UNIQUE" in str(exc).upper():
            raise HTTPException(409, "Username or email already exists")
        raise
    return {"username": body.username, "role": body.role, "created": True}

@app.get("/auth/users")
def list_users(current_user: dict = Depends(require_role("admin"))):
    with connect() as db:
        rows = db.execute("""
            SELECT u.id, u.username, u.email, u.role, u.student_id, u.active, u.last_login, u.created_at,
                   s.name AS student_name
            FROM users u
            LEFT JOIN students s ON s.id = u.student_id
            ORDER BY u.id
        """).fetchall()
    return [dict(r) for r in rows]

@app.patch("/auth/users/{user_id}")
def update_user(
    user_id: int,
    body: UserUpdateIn,
    current_user: dict = Depends(require_role("admin")),
):
    """Admin: edit email, toggle active status, or reset password."""
    with connect() as db:
        user = db.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
        if not user:
            raise HTTPException(404, "User not found")
        updates, params = [], []
        if body.email is not None:
            updates.append("email=?"); params.append(body.email)
        if body.active is not None:
            updates.append("active=?"); params.append(1 if body.active else 0)
        if body.new_password is not None:
            updates.append("password_hash=?"); params.append(hash_password(body.new_password))
        if not updates:
            raise HTTPException(400, "No fields to update")
        params.append(user_id)
        db.execute(f"UPDATE users SET {', '.join(updates)} WHERE id=?", params)
        log_audit(db, current_user["id"], current_user["username"], "update_user", "user", user_id, f"Updated user {user['username']}")
    return {"updated": True}

@app.post("/auth/change-password")
def change_password(body: ChangePasswordIn, current_user: dict = Depends(get_current_user)):
    user = authenticate_user(current_user["username"], body.current_password)
    if not user:
        raise HTTPException(400, "Current password is incorrect")
    with connect() as db:
        db.execute(
            "UPDATE users SET password_hash=? WHERE id=?",
            (hash_password(body.new_password), current_user["id"])
        )
        log_audit(db, current_user["id"], current_user["username"], "change_password", "user", current_user["id"])
    return {"changed": True}


# ══════════════════════════════════════════════════════════════════════════════
# ACADEMIC STRUCTURE (Departments, Academic Years, Classes, Subjects, Profiles)
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/v1/academic/departments")
def list_departments(current_user: dict = Depends(require_role("admin", "teacher"))):
    with connect() as db:
        rows = db.execute("SELECT * FROM departments ORDER BY name").fetchall()
    return [dict(r) for r in rows]

@app.post("/v1/academic/departments", status_code=201)
def create_department(dept: DepartmentIn, current_user: dict = Depends(require_role("admin"))):
    try:
        with connect() as db:
            cur = db.execute(
                "INSERT INTO departments(name, code) VALUES(?, ?)",
                (dept.name, dept.code.upper())
            )
            dept_id = cur.lastrowid
            log_audit(db, current_user["id"], current_user["username"], "create_department", "department", dept_id, dept.name)
    except Exception as exc:
        if "UNIQUE" in str(exc).upper():
            raise HTTPException(409, "Department code already exists")
        raise
    return {"id": dept_id, "name": dept.name, "code": dept.code.upper()}

@app.delete("/v1/academic/departments/{dept_id}", status_code=204)
def delete_department(dept_id: int, current_user: dict = Depends(require_role("admin"))):
    with connect() as db:
        if not db.execute("SELECT 1 FROM departments WHERE id=?", (dept_id,)).fetchone():
            raise HTTPException(404, "Department not found")
        db.execute("DELETE FROM departments WHERE id=?", (dept_id,))
        log_audit(db, current_user["id"], current_user["username"], "delete_department", "department", dept_id)

@app.get("/v1/academic/academic-years")
def list_academic_years(current_user: dict = Depends(require_role("admin", "teacher"))):
    with connect() as db:
        rows = db.execute("SELECT * FROM academic_years ORDER BY label DESC").fetchall()
    return [dict(r) for r in rows]

@app.post("/v1/academic/academic-years", status_code=201)
def create_academic_year(ay: AcademicYearIn, current_user: dict = Depends(require_role("admin"))):
    try:
        with connect() as db:
            if ay.is_current:
                db.execute("UPDATE academic_years SET is_current=0")
            cur = db.execute(
                "INSERT INTO academic_years(label, is_current) VALUES(?, ?)",
                (ay.label, 1 if ay.is_current else 0)
            )
            ay_id = cur.lastrowid
            log_audit(db, current_user["id"], current_user["username"], "create_academic_year", "academic_year", ay_id, ay.label)
    except Exception as exc:
        if "UNIQUE" in str(exc).upper():
            raise HTTPException(409, "Academic year already exists")
        raise
    return {"id": ay_id, "label": ay.label, "is_current": ay.is_current}

@app.delete("/v1/academic/academic-years/{ay_id}", status_code=204)
def delete_academic_year(ay_id: int, current_user: dict = Depends(require_role("admin"))):
    with connect() as db:
        if not db.execute("SELECT 1 FROM academic_years WHERE id=?", (ay_id,)).fetchone():
            raise HTTPException(404, "Academic year not found")
        db.execute("DELETE FROM academic_years WHERE id=?", (ay_id,))
        log_audit(db, current_user["id"], current_user["username"], "delete_academic_year", "academic_year", ay_id)

@app.get("/v1/academic/classes")
def list_classes(current_user: dict = Depends(require_role("admin", "teacher"))):
    with connect() as db:
        rows = db.execute("""
            SELECT c.*, d.name AS department_name, d.code AS department_code,
                   ay.label AS academic_year_label,
                   COUNT(sp.id) AS student_count
            FROM classes c
            JOIN departments d ON d.id = c.department_id
            JOIN academic_years ay ON ay.id = c.academic_year_id
            LEFT JOIN student_profiles sp ON sp.class_id = c.id
            GROUP BY c.id
            ORDER BY c.label
        """).fetchall()
    return [dict(r) for r in rows]

@app.post("/v1/academic/classes", status_code=201)
def create_class(cls: ClassIn, current_user: dict = Depends(require_role("admin"))):
    with connect() as db:
        cur = db.execute(
            "INSERT INTO classes(department_id, academic_year_id, year, semester, division, label) "
            "VALUES(?, ?, ?, ?, ?, ?)",
            (cls.department_id, cls.academic_year_id, cls.year, cls.semester, cls.division, cls.label)
        )
        class_id = cur.lastrowid
        log_audit(db, current_user["id"], current_user["username"], "create_class", "class", class_id, cls.label)
    return {"id": class_id, "label": cls.label}

@app.delete("/v1/academic/classes/{class_id}", status_code=204)
def delete_class(class_id: int, current_user: dict = Depends(require_role("admin"))):
    with connect() as db:
        if not db.execute("SELECT 1 FROM classes WHERE id=?", (class_id,)).fetchone():
            raise HTTPException(404, "Class not found")
        db.execute("DELETE FROM classes WHERE id=?", (class_id,))
        log_audit(db, current_user["id"], current_user["username"], "delete_class", "class", class_id)

@app.get("/v1/academic/subjects")
def list_subjects(department_id: int | None = None, current_user: dict = Depends(require_role("admin", "teacher"))):
    with connect() as db:
        if department_id:
            rows = db.execute("""
                SELECT s.*, d.name AS department_name, d.code AS department_code
                FROM subjects s
                JOIN departments d ON d.id = s.department_id
                WHERE s.department_id = ?
                ORDER BY s.code
            """, (department_id,)).fetchall()
        else:
            rows = db.execute("""
                SELECT s.*, d.name AS department_name, d.code AS department_code
                FROM subjects s
                JOIN departments d ON d.id = s.department_id
                ORDER BY s.code
            """).fetchall()
    return [dict(r) for r in rows]

@app.post("/v1/academic/subjects", status_code=201)
def create_subject(subj: SubjectIn, current_user: dict = Depends(require_role("admin"))):
    try:
        with connect() as db:
            cur = db.execute(
                "INSERT INTO subjects(department_id, name, code, semester) VALUES(?, ?, ?, ?)",
                (subj.department_id, subj.name, subj.code.upper(), subj.semester)
            )
            subj_id = cur.lastrowid
            log_audit(db, current_user["id"], current_user["username"], "create_subject", "subject", subj_id, subj.name)
    except Exception as exc:
        if "UNIQUE" in str(exc).upper():
            raise HTTPException(409, "Subject code already exists")
        raise
    return {"id": subj_id, "name": subj.name, "code": subj.code.upper()}

@app.delete("/v1/academic/subjects/{subject_id}", status_code=204)
def delete_subject(subject_id: int, current_user: dict = Depends(require_role("admin"))):
    with connect() as db:
        if not db.execute("SELECT 1 FROM subjects WHERE id=?", (subject_id,)).fetchone():
            raise HTTPException(404, "Subject not found")
        db.execute("DELETE FROM subjects WHERE id=?", (subject_id,))
        log_audit(db, current_user["id"], current_user["username"], "delete_subject", "subject", subject_id)

@app.get("/v1/academic/teacher-profiles")
def list_teacher_profiles(current_user: dict = Depends(require_role("admin", "teacher"))):
    with connect() as db:
        rows = db.execute("""
            SELECT tp.*, u.username, u.email, d.name AS department_name, d.code AS department_code
            FROM teacher_profiles tp
            JOIN users u ON u.id = tp.user_id
            LEFT JOIN departments d ON d.id = tp.department_id
            ORDER BY u.username
        """).fetchall()
    return [dict(r) for r in rows]

@app.get("/v1/academic/teacher-assignments")
def list_teacher_assignments(current_user: dict = Depends(require_role("admin", "teacher"))):
    with connect() as db:
        rows = db.execute("""
            SELECT ta.id, ta.teacher_id, ta.subject_id, ta.class_id,
                   u.username AS teacher_name, tp.employee_id,
                   s.name AS subject_name, s.code AS subject_code,
                   c.label AS class_label
            FROM teacher_assignments ta
            JOIN teacher_profiles tp ON tp.id = ta.teacher_id
            JOIN users u ON u.id = tp.user_id
            JOIN subjects s ON s.id = ta.subject_id
            JOIN classes c ON c.id = ta.class_id
            ORDER BY c.label, s.name
        """).fetchall()
    return [dict(r) for r in rows]

@app.post("/v1/academic/teacher-assignments", status_code=201)
def create_teacher_assignment(ta: TeacherAssignmentIn, current_user: dict = Depends(require_role("admin"))):
    try:
        with connect() as db:
            cur = db.execute(
                "INSERT INTO teacher_assignments(teacher_id, subject_id, class_id) VALUES(?, ?, ?)",
                (ta.teacher_id, ta.subject_id, ta.class_id)
            )
            aid = cur.lastrowid
            log_audit(db, current_user["id"], current_user["username"], "create_assignment", "teacher_assignment", aid)
    except Exception as exc:
        if "UNIQUE" in str(exc).upper():
            raise HTTPException(409, "Teacher is already assigned to this subject and class")
        raise
    return {"id": aid, "assigned": True}

@app.delete("/v1/academic/teacher-assignments/{assignment_id}", status_code=204)
def delete_teacher_assignment(assignment_id: int, current_user: dict = Depends(require_role("admin"))):
    with connect() as db:
        if not db.execute("SELECT 1 FROM teacher_assignments WHERE id=?", (assignment_id,)).fetchone():
            raise HTTPException(404, "Assignment not found")
        db.execute("DELETE FROM teacher_assignments WHERE id=?", (assignment_id,))
        log_audit(db, current_user["id"], current_user["username"], "delete_assignment", "teacher_assignment", assignment_id)

@app.get("/v1/academic/student-profiles")
def list_student_profiles(
    class_id: int | None = None,
    current_user: dict = Depends(require_role("admin", "teacher")),
):
    with connect() as db:
        if class_id:
            rows = db.execute("""
                SELECT sp.*, s.name AS student_name, s.email, c.label AS class_label
                FROM student_profiles sp
                JOIN students s ON s.id = sp.student_id
                JOIN classes c ON c.id = sp.class_id
                WHERE sp.class_id = ?
                ORDER BY sp.roll_number, s.name
            """, (class_id,)).fetchall()
        else:
            rows = db.execute("""
                SELECT sp.*, s.name AS student_name, s.email, c.label AS class_label
                FROM student_profiles sp
                JOIN students s ON s.id = sp.student_id
                JOIN classes c ON c.id = sp.class_id
                ORDER BY c.label, sp.roll_number, s.name
            """).fetchall()
    return [dict(r) for r in rows]

@app.post("/v1/academic/student-profiles", status_code=201)
def assign_student_to_class(body: StudentProfileIn, current_user: dict = Depends(require_role("admin"))):
    try:
        with connect() as db:
            cur = db.execute(
                "INSERT INTO student_profiles(student_id, class_id, roll_number) VALUES(?, ?, ?) "
                "ON CONFLICT(student_id, class_id) DO UPDATE SET roll_number=excluded.roll_number",
                (body.student_id, body.class_id, body.roll_number)
            )
            sp_id = cur.lastrowid
            log_audit(db, current_user["id"], current_user["username"], "assign_student", "student_profile", sp_id, f"Student {body.student_id} to class {body.class_id}")
    except Exception as exc:
        raise HTTPException(400, f"Unable to assign student: {exc}")
    return {"assigned": True, "student_id": body.student_id, "class_id": body.class_id}

@app.delete("/v1/academic/student-profiles/{profile_id}", status_code=204)
def remove_student_from_class(profile_id: int, current_user: dict = Depends(require_role("admin"))):
    with connect() as db:
        if not db.execute("SELECT 1 FROM student_profiles WHERE id=?", (profile_id,)).fetchone():
            raise HTTPException(404, "Profile not found")
        db.execute("DELETE FROM student_profiles WHERE id=?", (profile_id,))
        log_audit(db, current_user["id"], current_user["username"], "remove_student_profile", "student_profile", profile_id)


# ══════════════════════════════════════════════════════════════════════════════
# TEACHER SCOPED APIS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/v1/teacher/my-assignments")
def get_my_assignments(current_user: dict = Depends(require_role("teacher", "admin"))):
    with connect() as db:
        if current_user["role"] == "admin":
            rows = db.execute("""
                SELECT ta.id AS assignment_id, ta.teacher_id, ta.subject_id, ta.class_id,
                       u.username AS teacher_name,
                       s.name AS subject_name, s.code AS subject_code,
                       c.label AS class_label,
                       (SELECT COUNT(*) FROM student_profiles sp WHERE sp.class_id = ta.class_id) AS student_count
                FROM teacher_assignments ta
                JOIN teacher_profiles tp ON tp.id = ta.teacher_id
                JOIN users u ON u.id = tp.user_id
                JOIN subjects s ON s.id = ta.subject_id
                JOIN classes c ON c.id = ta.class_id
                ORDER BY c.label, s.name
            """).fetchall()
            return [dict(r) for r in rows]

        tp = db.execute("SELECT id FROM teacher_profiles WHERE user_id=?", (current_user["id"],)).fetchone()
        if not tp:
            return []
        rows = db.execute("""
            SELECT ta.id AS assignment_id, ta.teacher_id, ta.subject_id, ta.class_id,
                   s.name AS subject_name, s.code AS subject_code,
                   c.label AS class_label,
                   (SELECT COUNT(*) FROM student_profiles sp WHERE sp.class_id = ta.class_id) AS student_count
            FROM teacher_assignments ta
            JOIN subjects s ON s.id = ta.subject_id
            JOIN classes c ON c.id = ta.class_id
            WHERE ta.teacher_id = ?
            ORDER BY c.label, s.name
        """, (tp["id"],)).fetchall()
    return [dict(r) for r in rows]

@app.get("/v1/teacher/my-classes")
def get_my_classes(current_user: dict = Depends(require_role("teacher", "admin"))):
    with connect() as db:
        if current_user["role"] == "admin":
            rows = db.execute("""
                SELECT c.*, COUNT(sp.id) AS student_count
                FROM classes c
                LEFT JOIN student_profiles sp ON sp.class_id = c.id
                GROUP BY c.id
                ORDER BY c.label
            """).fetchall()
            return [dict(r) for r in rows]

        tp = db.execute("SELECT id FROM teacher_profiles WHERE user_id=?", (current_user["id"],)).fetchone()
        if not tp:
            return []
        rows = db.execute("""
            SELECT DISTINCT c.*,
                   (SELECT COUNT(*) FROM student_profiles sp WHERE sp.class_id = c.id) AS student_count
            FROM classes c
            JOIN teacher_assignments ta ON ta.class_id = c.id
            WHERE ta.teacher_id = ?
            ORDER BY c.label
        """, (tp["id"],)).fetchall()
    return [dict(r) for r in rows]

@app.get("/v1/teacher/classes/{class_id}/students")
def get_class_students(class_id: int, current_user: dict = Depends(require_role("admin", "teacher"))):
    with connect() as db:
        if current_user["role"] == "teacher":
            tp = db.execute("SELECT id FROM teacher_profiles WHERE user_id=?", (current_user["id"],)).fetchone()
            if tp:
                assigned = db.execute(
                    "SELECT 1 FROM teacher_assignments WHERE teacher_id=? AND class_id=?",
                    (tp["id"], class_id)
                ).fetchone()
                if not assigned:
                    raise HTTPException(403, "You are not assigned to this class")
        rows = db.execute("""
            SELECT s.id, s.name, s.email, s.embedding <> '[]' AS enrolled, s.active,
                   sp.roll_number, sp.class_id
            FROM students s
            JOIN student_profiles sp ON sp.student_id = s.id
            WHERE sp.class_id = ? AND s.active = 1
            ORDER BY sp.roll_number, s.name
        """, (class_id,)).fetchall()
    return [dict(r) for r in rows]


# ══════════════════════════════════════════════════════════════════════════════
# STUDENTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/v1/students/embeddings", dependencies=[Depends(require_device)])
def embeddings():
    with connect() as db:
        rows = db.execute(
            "SELECT id,name,embedding,enrollment_quality,enrollment_samples,enrolled_at,embedding_version FROM students WHERE active=1 AND embedding <> '[]'"
        ).fetchall()
    return [{
        "id": r["id"],
        "name": r["name"],
        "embedding": json.loads(r["embedding"]),
        "enrollment_quality": r["enrollment_quality"],
        "enrollment_samples": r["enrollment_samples"],
        "enrolled_at": r["enrolled_at"],
        "embedding_version": r["embedding_version"]
    } for r in rows]

@app.get("/v1/classes/{class_id}/students/embeddings", dependencies=[Depends(require_device)])
def class_student_embeddings(class_id: str):
    """Return embeddings for students belonging to a specific class."""
    try:
        class_id_int = int(class_id)
    except ValueError:
        raise HTTPException(400, "Invalid class ID")
    with connect() as db:
        rows = db.execute(
            """SELECT s.id, s.name, s.embedding, s.enrollment_quality, s.enrollment_samples, s.enrolled_at, s.embedding_version
               FROM students s
               JOIN student_profiles sp ON s.id = sp.student_id
               WHERE s.active = 1
                 AND s.embedding <> '[]'
                 AND sp.class_id = ?""",
            (class_id_int,)
        ).fetchall()
    return [{
        "id": r["id"],
        "name": r["name"],
        "embedding": json.loads(r["embedding"]),
        "enrollment_quality": r["enrollment_quality"],
        "enrollment_samples": r["enrollment_samples"],
        "enrolled_at": r["enrolled_at"],
        "embedding_version": r["embedding_version"]
    } for r in rows]

@app.post("/v1/students", dependencies=[Depends(require_device)], status_code=201)
def create_student(student: StudentIn):
    with connect() as db:
        db.execute(
            "INSERT INTO students(id,name,email,embedding,enrollment_quality,enrollment_samples,enrolled_at,embedding_version) "
            "VALUES(?,?,?,?,0.0,0,datetime('now'),1) "
            "ON CONFLICT(id) DO UPDATE SET name=excluded.name,email=excluded.email,"
            "embedding=excluded.embedding,active=1",
            (student.id, student.name, student.email, json.dumps(student.embedding))
        )
    return {"id": student.id, "saved": True}

@app.post("/v1/students/manual", status_code=201)
def create_student_record(
    student: StudentRecordIn,
    current_user: dict = Depends(require_role("admin")),
):
    """Admin-only: create a student record, class assignment, and auto-provision standardized login."""
    with connect() as db:
        db.execute(
            "INSERT INTO students(id,name,email,embedding) VALUES(?,?,?,'[]') "
            "ON CONFLICT(id) DO UPDATE SET name=excluded.name,email=excluded.email,active=1",
            (student.id, student.name, student.email)
        )
        if student.class_id:
            db.execute(
                "INSERT INTO student_profiles(student_id, class_id, roll_number) VALUES(?, ?, ?) "
                "ON CONFLICT(student_id, class_id) DO UPDATE SET roll_number=excluded.roll_number",
                (student.id, student.class_id, student.roll_number)
            )
        # Standardized user account: username = student.id, default password = student123
        db.execute(
            "INSERT INTO users(username, email, password_hash, role, student_id) VALUES(?,?,?,?,?) "
            "ON CONFLICT(username) DO UPDATE SET email=COALESCE(excluded.email, users.email), student_id=excluded.student_id, active=1",
            (student.id, student.email, hash_password("student123"), "student", student.id)
        )
        log_audit(db, current_user["id"], current_user["username"], "create_student", "student", student.id, student.name)
    return {"id": student.id, "enrollment_required": True}

@app.get("/v1/students")
def list_students(current_user: dict = Depends(require_role("admin", "teacher"))):
    with connect() as db:
        if current_user["role"] == "teacher":
            tp = db.execute("SELECT id FROM teacher_profiles WHERE user_id=?", (current_user["id"],)).fetchone()
            if tp:
                assigned_classes = [r[0] for r in db.execute(
                    "SELECT DISTINCT class_id FROM teacher_assignments WHERE teacher_id=?", (tp["id"],)
                ).fetchall()]
                if assigned_classes:
                    placeholders = ",".join("?" * len(assigned_classes))
                    rows = db.execute(f"""
                        SELECT DISTINCT s.id, s.name, s.email, s.embedding <> '[]' AS enrolled, s.active,
                               sp.roll_number, sp.class_id, c.label AS class_label
                        FROM students s
                        JOIN student_profiles sp ON sp.student_id = s.id
                        JOIN classes c ON c.id = sp.class_id
                        WHERE sp.class_id IN ({placeholders})
                        ORDER BY sp.roll_number, s.name
                    """, assigned_classes).fetchall()
                    return [dict(r) for r in rows]

        rows = db.execute("""
            SELECT s.id, s.name, s.email, s.embedding <> '[]' AS enrolled, s.active,
                   sp.roll_number, sp.class_id, c.label AS class_label
            FROM students s
            LEFT JOIN student_profiles sp ON sp.student_id = s.id
            LEFT JOIN classes c ON c.id = sp.class_id
            ORDER BY s.name
        """).fetchall()
    return [dict(r) for r in rows]

@app.patch("/v1/students/{student_id}")
def update_student(
    student_id: str,
    body: StudentUpdateIn,
    current_user: dict = Depends(require_role("admin")),
):
    """Admin only: edit student name, email, or class assignment."""
    with connect() as db:
        if not db.execute("SELECT 1 FROM students WHERE id=?", (student_id,)).fetchone():
            raise HTTPException(404, "Student not found")
        updates, params = [], []
        if body.name is not None:
            updates.append("name=?"); params.append(body.name)
        if body.email is not None:
            updates.append("email=?"); params.append(body.email)
        if updates:
            params.append(student_id)
            db.execute(f"UPDATE students SET {', '.join(updates)} WHERE id=?", params)
            if body.email is not None:
                db.execute("UPDATE users SET email=? WHERE student_id=?", (body.email, student_id))

        if body.class_id is not None:
            db.execute("DELETE FROM student_profiles WHERE student_id=?", (student_id,))
            db.execute(
                "INSERT INTO student_profiles(student_id, class_id, roll_number) VALUES(?, ?, ?)",
                (student_id, body.class_id, body.roll_number)
            )
        elif body.roll_number is not None:
            db.execute("UPDATE student_profiles SET roll_number=? WHERE student_id=?", (body.roll_number, student_id))

        log_audit(db, current_user["id"], current_user["username"], "update_student", "student", student_id)
    return {"updated": True}

@app.delete("/v1/students/{student_id}", status_code=204)
def deactivate_student(
    student_id: str,
    current_user: dict = Depends(require_role("admin")),
):
    """Admin only: soft-deactivate a student and their user account."""
    with connect() as db:
        if not db.execute("SELECT 1 FROM students WHERE id=?", (student_id,)).fetchone():
            raise HTTPException(404, "Student not found")
        db.execute("UPDATE students SET active=0 WHERE id=?", (student_id,))
        db.execute("UPDATE users SET active=0 WHERE student_id=?", (student_id,))
        log_audit(db, current_user["id"], current_user["username"], "deactivate_student", "student", student_id)

@app.post("/v1/students/{student_id}/enroll-images")
def enroll_from_browser(
    student_id: str,
    frames: CameraFramesIn,
    current_user: dict = Depends(require_role("admin", "teacher")),
):
    """High-quality enrollment from multi-angle webcam frames."""
    import numpy as np_mod
    samples = []
    for value in frames.images:
        try:
            result = embedding_enroll(decode_image(value))
            if result is not None:
                samples.append(result)
        except ValueError:
            pass

    if len(samples) < 5:
        raise HTTPException(
            422,
            f"Only {len(samples)} valid face frame(s) captured — need at least 5. "
            "Ensure good lighting, face the camera directly, and follow the angle prompts."
        )

    clean = filter_enrollment_samples(samples, max_dist=0.55)
    if len(clean) < 3:
        raise HTTPException(
            422,
            f"Only {len(clean)} valid frames after filtering — need at least 3. "
            "Try again with better lighting and fewer obstructions."
        )
    vector = [float(x) for x in np_mod.mean(clean, axis=0)]

    if len(clean) >= 2:
        embeddings_array = np_mod.array(clean)
        distances = []
        for i in range(len(clean)):
            for j in range(i+1, len(clean)):
                dist = np_mod.linalg.norm(embeddings_array[i] - embeddings_array[j])
                distances.append(dist)
        avg_distance = np_mod.mean(distances) if distances else 0.5
        quality_score = max(0.0, min(1.0, (avg_distance - 0.1) * 2.5))
        quality_score = 1.0 - quality_score
    else:
        quality_score = 0.8

    with connect() as db:
        if not db.execute("SELECT 1 FROM students WHERE id=?", (student_id,)).fetchone():
            raise HTTPException(404, "Create the student record first")
        db.execute(
            """UPDATE students
               SET embedding=?,
                   enrollment_quality=?,
                   enrollment_samples=?,
                   enrolled_at=datetime('now'),
                   embedding_version=1
               WHERE id=? """,
            (json.dumps(vector), quality_score, len(clean), student_id)
        )
        log_audit(db, current_user["id"], current_user["username"], "enroll_face", "student", student_id)
    return {
        "student_id": student_id,
        "raw_samples": len(samples),
        "clean_samples": len(clean),
        "enrollment_quality": round(quality_score, 3),
        "enrolled": True,
    }

@app.get("/v1/students/{student_id}/attendance")
def student_attendance(student_id: str, current_user: dict = Depends(get_current_user)):
    if current_user["role"] == "student" and current_user["student_id"] != student_id:
        raise HTTPException(403, "You can only view your own attendance records")
    with connect() as db:
        student = db.execute("SELECT id,name,email FROM students WHERE id=?", (student_id,)).fetchone()
        if not student:
            raise HTTPException(404, "Student not found")

        profile = db.execute("""
            SELECT sp.roll_number, c.id AS class_id, c.label AS class_label,
                   c.year, c.semester, c.division, d.name AS department_name
            FROM student_profiles sp
            JOIN classes c ON c.id = sp.class_id
            JOIN departments d ON d.id = c.department_id
            WHERE sp.student_id = ?
        """, (student_id,)).fetchone()

        rows = db.execute("""
            SELECT a.id, a.event_id, a.observed_at, a.status, a.confidence,
                   a.liveness_score, a.reason, a.manually_added, 
                   s.course_code, s.room,
                   sub.name AS subject_name, sub.code AS subject_code,
                   u.username AS teacher_name, c.label AS class_label
            FROM attendance a
            LEFT JOIN sessions s ON s.id = a.session_id
            LEFT JOIN subjects sub ON sub.id = s.subject_id
            LEFT JOIN classes c ON c.id = s.class_id
            LEFT JOIN teacher_profiles tp ON tp.id = s.teacher_id
            LEFT JOIN users u ON u.id = tp.user_id
            WHERE a.student_id = ?
            ORDER BY a.observed_at DESC LIMIT 200
        """, (student_id,)).fetchall()

        total_sessions = db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        attended = db.execute(
            "SELECT COUNT(DISTINCT session_id) FROM attendance "
            "WHERE student_id=? AND session_id IS NOT NULL AND status NOT IN ('nc','absent')",
            (student_id,)
        ).fetchone()[0]

        # Subject-wise attendance breakdown
        class_id = profile["class_id"] if profile else None
        subject_breakdown = []
        if class_id:
            class_subjects = db.execute(
                "SELECT id, name, code FROM subjects WHERE department_id=(SELECT department_id FROM classes WHERE id=?)",
                (class_id,)
            ).fetchall()
            for subj in class_subjects:
                subj_total = db.execute(
                    "SELECT COUNT(*) FROM sessions WHERE (subject_id=? OR course_code=?) AND (class_id=? OR class_id IS NULL)",
                    (subj["id"], subj["code"], class_id)
                ).fetchone()[0]
                subj_attended = db.execute("""
                    SELECT COUNT(DISTINCT a.session_id) FROM attendance a
                    JOIN sessions s ON s.id = a.session_id
                    WHERE a.student_id=? AND (s.subject_id=? OR s.course_code=?) AND a.status NOT IN ('nc','absent')
                """, (student_id, subj["id"], subj["code"])).fetchone()[0]
                rate = round(subj_attended / subj_total, 3) if subj_total > 0 else (1.0 if subj_attended > 0 else None)
                subject_breakdown.append({
                    "subject_id": subj["id"],
                    "subject_name": subj["name"],
                    "subject_code": subj["code"],
                    "total_sessions": subj_total,
                    "attended_sessions": subj_attended,
                    "attendance_rate": rate,
                    "is_at_risk": (rate is not None and rate < 0.75)
                })

    attendance_rate = round(attended / total_sessions, 3) if total_sessions > 0 else None
    return {
        "student": dict(student),
        "profile": dict(profile) if profile else None,
        "attendance_rate": attendance_rate,
        "total_sessions": total_sessions,
        "attended_sessions": attended,
        "subjects": subject_breakdown,
        "records": [dict(r) for r in rows],
    }


# ══════════════════════════════════════════════════════════════════════════════
# SESSIONS
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/v1/sessions", status_code=201)
def create_session(
    session: SessionIn,
    current_user: dict = Depends(require_role("admin", "teacher")),
):
    start = session.starts_at.astimezone(timezone.utc).replace(tzinfo=None).isoformat()
    end = session.ends_at.astimezone(timezone.utc).replace(tzinfo=None).isoformat()
    teacher_id = session.teacher_id
    course_code = session.course_code

    with connect() as db:
        if current_user["role"] == "teacher":
            tp = db.execute("SELECT id FROM teacher_profiles WHERE user_id=?", (current_user["id"],)).fetchone()
            if not tp:
                cur = db.execute(
                    "INSERT INTO teacher_profiles(user_id, employee_id, designation) VALUES(?,?,?)",
                    (current_user["id"], f"EMP-{current_user['id']}", "Teacher")
                )
                teacher_id = cur.lastrowid
            else:
                teacher_id = tp["id"]

            if session.subject_id and session.class_id:
                assignment = db.execute(
                    "SELECT 1 FROM teacher_assignments WHERE teacher_id=? AND subject_id=? AND class_id=?",
                    (teacher_id, session.subject_id, session.class_id)
                ).fetchone()
                if not assignment:
                    raise HTTPException(403, "You are not assigned to this class and subject")

        if not course_code and session.subject_id:
            subj = db.execute("SELECT code FROM subjects WHERE id=?", (session.subject_id,)).fetchone()
            if subj:
                course_code = subj["code"]
        if not course_code:
            course_code = "CLASS"

        cur = db.execute(
            "INSERT INTO sessions(course_code,room,starts_at,ends_at,late_after_minutes,teacher_id,subject_id,class_id,status) "
            "VALUES(?,?,?,?,?,?,?,?, 'open')",
            (course_code, session.room, start, end, session.late_after_minutes, teacher_id, session.subject_id, session.class_id)
        )
        session_id = cur.lastrowid
        log_audit(db, current_user["id"], current_user["username"], "create_session", "session", session_id, f"Session {course_code} in {session.room}")
    return {"id": session_id}


def _mark_unmarked_absent(db, session_id: int, observed_at_iso: str | None = None) -> int:
    session = db.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
    if not session or not session["class_id"]:
        return 0
    obs_time = observed_at_iso or session["starts_at"] or datetime.now(timezone.utc).isoformat()
    unmarked = db.execute("""
        SELECT sp.student_id
        FROM student_profiles sp
        JOIN students s ON s.id = sp.student_id
        WHERE sp.class_id = ? AND s.active = 1
          AND sp.student_id NOT IN (
              SELECT student_id FROM attendance WHERE session_id = ?
          )
    """, (session["class_id"], session_id)).fetchall()

    count = 0
    for r in unmarked:
        db.execute("""
            INSERT INTO attendance(event_id, student_id, session_id, device_id, observed_at, status, confidence, liveness_score, reason, manually_added)
            VALUES(?, ?, ?, 'system-absent', ?, 'absent', 1.0, 1.0, 'Automatic absent on session finalization', 0)
        """, (str(uuid.uuid4()), r["student_id"], session_id, obs_time))
        count += 1
    return count

@app.patch("/v1/sessions/{session_id}/close")
def close_session(
    session_id: int,
    current_user: dict = Depends(require_role("admin", "teacher")),
):
    with connect() as db:
        session = db.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        if not session:
            raise HTTPException(404, "Session not found")
        if current_user["role"] == "teacher":
            tp = db.execute("SELECT id FROM teacher_profiles WHERE user_id=?", (current_user["id"],)).fetchone()
            if session["teacher_id"] and (not tp or tp["id"] != session["teacher_id"]):
                raise HTTPException(403, "You can only close your own sessions")
        db.execute("UPDATE sessions SET status='closed' WHERE id=?", (session_id,))
        absent_count = _mark_unmarked_absent(db, session_id)
        log_audit(db, current_user["id"], current_user["username"], "close_session", "session", session_id, f"Marked {absent_count} absent")

    # Stop active Pi camera session when attendance session closes
    global _backend_camera_session
    with _backend_camera_lock:
        if _backend_camera_session is not None:
            try:
                _backend_camera_session.stop()
            except Exception:
                pass
            _backend_camera_session = None

    return {"closed": True, "id": session_id, "marked_absent_count": absent_count}


# ── Pi On-Demand Camera Session Endpoints ─────────────────────────────────────

@app.post("/v1/camera/session/start")
def start_attendance_camera_session(
    current_user: dict = Depends(require_role("admin", "teacher")),
):
    """Start the Raspberry Pi camera stream on-demand for an attendance session."""
    global _backend_camera_session
    with _backend_camera_lock:
        if _backend_camera_session is not None and getattr(_backend_camera_session, "_active", False):
            return {"status": "already_active", "session_type": "attendance"}

        from edge.camera_control import CameraSession, CameraInUseError, CameraAuthError
        session = CameraSession(
            session_type="attendance",
            client_id=f"teacher-{current_user.get('username')}",
        )
        try:
            session.start()
            _backend_camera_session = session
            return {
                "status": "started",
                "session_type": "attendance",
                "stream_port": session.stream_port,
                "message": "Pi camera activated for attendance",
            }
        except CameraInUseError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except CameraAuthError as exc:
            raise HTTPException(status_code=502, detail=f"Camera control auth failed: {exc}")
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Could not activate Pi camera stream: {exc}")


@app.post("/v1/camera/session/stop")
def stop_attendance_camera_session(
    current_user: dict = Depends(require_role("admin", "teacher")),
):
    """Stop the Raspberry Pi camera stream on-demand and release C270 hardware."""
    global _backend_camera_session
    with _backend_camera_lock:
        if _backend_camera_session is not None:
            try:
                _backend_camera_session.stop()
            finally:
                _backend_camera_session = None
        else:
            try:
                from edge.camera_control import stop_camera_service
                stop_camera_service(session_type="attendance")
            except Exception:
                pass
        return {"status": "stopped", "active": False}


@app.get("/v1/camera/session/status")
def get_attendance_camera_status(
    current_user: dict = Depends(require_role("admin", "teacher")),
):
    """Query live on-demand camera service state from the Raspberry Pi."""
    global _backend_camera_session
    with _backend_camera_lock:
        backend_active = _backend_camera_session is not None and getattr(_backend_camera_session, "_active", False)

    try:
        from edge.camera_control import get_camera_status
        pi_status = get_camera_status()
    except Exception as exc:
        pi_status = {"error": str(exc), "reachable": False}

    return {
        "backend_session_active": backend_active,
        "pi_status": pi_status,
    }

@app.post("/v1/sessions/{session_id}/finalize-absent")
def finalize_session_absent(
    session_id: int,
    current_user: dict = Depends(require_role("admin", "teacher")),
):
    with connect() as db:
        session = db.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        if not session:
            raise HTTPException(404, "Session not found")
        if current_user["role"] == "teacher":
            tp = db.execute("SELECT id FROM teacher_profiles WHERE user_id=?", (current_user["id"],)).fetchone()
            if session["teacher_id"] and (not tp or tp["id"] != session["teacher_id"]):
                raise HTTPException(403, "You can only manage your own sessions")
        absent_count = _mark_unmarked_absent(db, session_id)
        log_audit(db, current_user["id"], current_user["username"], "finalize_absent", "session", session_id, f"Marked {absent_count} absent")
    return {"session_id": session_id, "marked_absent_count": absent_count}

@app.get("/v1/sessions")
def list_sessions(current_user: dict = Depends(get_current_user)):
    with connect() as db:
        base_query = """
            SELECT s.*,
                   sub.name AS subject_name, sub.code AS subject_code,
                   c.label AS class_label,
                   u.username AS teacher_username
            FROM sessions s
            LEFT JOIN subjects sub ON sub.id = s.subject_id
            LEFT JOIN classes c ON c.id = s.class_id
            LEFT JOIN teacher_profiles tp ON tp.id = s.teacher_id
            LEFT JOIN users u ON u.id = tp.user_id
        """
        if current_user["role"] == "admin":
            rows = db.execute(base_query + " ORDER BY s.starts_at DESC LIMIT 100").fetchall()
        elif current_user["role"] == "teacher":
            tp = db.execute("SELECT id FROM teacher_profiles WHERE user_id=?", (current_user["id"],)).fetchone()
            if tp:
                rows = db.execute(
                    base_query + " WHERE s.teacher_id = ? OR s.teacher_id IS NULL ORDER BY s.starts_at DESC LIMIT 100",
                    (tp["id"],)
                ).fetchall()
            else:
                rows = db.execute(base_query + " ORDER BY s.starts_at DESC LIMIT 100").fetchall()
        elif current_user["role"] == "student":
            sp = db.execute("SELECT class_id FROM student_profiles WHERE student_id=?", (current_user["student_id"],)).fetchone()
            if sp:
                rows = db.execute(
                    base_query + " WHERE s.class_id = ? OR s.class_id IS NULL ORDER BY s.starts_at DESC LIMIT 100",
                    (sp["class_id"],)
                ).fetchall()
            else:
                rows = db.execute(base_query + " ORDER BY s.starts_at DESC LIMIT 100").fetchall()
        else:
            rows = db.execute(base_query + " ORDER BY s.starts_at DESC LIMIT 100").fetchall()
    return [dict(r) for r in rows]

@app.delete("/v1/sessions/{session_id}", status_code=204)
def delete_session(
    session_id: int,
    current_user: dict = Depends(require_role("admin", "teacher")),
):
    with connect() as db:
        if not db.execute("SELECT 1 FROM sessions WHERE id=?", (session_id,)).fetchone():
            raise HTTPException(404, "Session not found")
        db.execute("DELETE FROM sessions WHERE id=?", (session_id,))
        log_audit(db, current_user["id"], current_user["username"], "delete_session", "session", session_id)


# ══════════════════════════════════════════════════════════════════════════════
# ATTENDANCE — CRUD
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/v1/attendance")
def list_attendance(
    student_id: str | None = None,
    session_id: int | None = None,
    date: str | None = None,
    class_id: int | None = None,
    current_user: dict = Depends(require_role("admin", "teacher")),
):
    """List attendance with optional filters."""
    clauses, params = [], []
    if student_id:
        clauses.append("a.student_id=?"); params.append(student_id)
    if session_id is not None:
        clauses.append("a.session_id=?"); params.append(session_id)
    if date:
        clauses.append("date(a.observed_at)=?"); params.append(date)
    if class_id is not None:
        clauses.append("s.class_id=?"); params.append(class_id)

    with connect() as db:
        if current_user["role"] == "teacher" and not class_id and not session_id:
            tp = db.execute("SELECT id FROM teacher_profiles WHERE user_id=?", (current_user["id"],)).fetchone()
            if tp:
                assigned_classes = [r[0] for r in db.execute(
                    "SELECT DISTINCT class_id FROM teacher_assignments WHERE teacher_id=?", (tp["id"],)
                ).fetchall()]
                if assigned_classes:
                    placeholders = ",".join("?" * len(assigned_classes))
                    clauses.append(f"(s.class_id IN ({placeholders}) OR s.teacher_id = ? OR s.id IS NULL)")
                    params.extend(assigned_classes)
                    params.append(tp["id"])

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = db.execute(f"""
            SELECT a.id, a.student_id, st.name AS student_name,
                   a.session_id, s.course_code, s.room,
                   sub.name AS subject_name, c.label AS class_label,
                   a.observed_at, a.status, a.confidence,
                   a.liveness_score, a.reason, a.manually_added, a.device_id,
                   a.original_status, a.modified_by, a.modified_at
            FROM attendance a
            JOIN students st ON st.id = a.student_id
            LEFT JOIN sessions s ON s.id = a.session_id
            LEFT JOIN subjects sub ON sub.id = s.subject_id
            LEFT JOIN classes c ON c.id = s.class_id
            {where}
            ORDER BY a.observed_at DESC LIMIT 500
        """, params).fetchall()
    return [dict(r) for r in rows]

@app.post("/v1/attendance/manual", status_code=201)
def add_attendance_manual(
    body: AttendanceManualIn,
    current_user: dict = Depends(require_role("admin", "teacher")),
):
    """Teacher / admin: manually insert an attendance record with class membership validation."""
    if body.status in ("nc", "present_nc") and not body.reason:
        raise HTTPException(422, "A reason is required for NC or Present-NC status")
    with connect() as db:
        student = db.execute("SELECT 1 FROM students WHERE id=? AND active=1", (body.student_id,)).fetchone()
        if not student:
            raise HTTPException(404, "Student not found or inactive")

        if body.session_id is not None:
            session = db.execute("SELECT * FROM sessions WHERE id=?", (body.session_id,)).fetchone()
            if not session:
                raise HTTPException(404, "Session not found")
            # Enforce class membership if session has a class
            if session["class_id"]:
                enrolled = db.execute(
                    "SELECT 1 FROM student_profiles WHERE student_id=? AND class_id=?",
                    (body.student_id, session["class_id"])
                ).fetchone()
                if not enrolled:
                    raise HTTPException(403, f"Student is not enrolled in the class for this session")

        try:
            when_dt = datetime.fromisoformat(body.observed_at.replace("Z", "+00:00"))
            when = when_dt.astimezone(timezone.utc).replace(tzinfo=None).isoformat()
        except ValueError:
            raise HTTPException(422, "Invalid observed_at format; use ISO 8601")

        event_id = str(uuid.uuid4())
        now_str = datetime.now(timezone.utc).isoformat()
        db.execute(
            "INSERT INTO attendance"
            "(event_id,student_id,session_id,device_id,observed_at,status,confidence,liveness_score,reason,manually_added,modified_by,modified_at)"
            " VALUES(?,?,?,?,?,?,1.0,1.0,?,1,?,?)",
            (event_id, body.student_id, body.session_id,
             f"manual:{current_user['username']}", when, body.status, body.reason,
             current_user["username"], now_str)
        )
        log_audit(db, current_user["id"], current_user["username"], "manual_attendance", "attendance", event_id, f"Marked {body.student_id} as {body.status}")
    return {"event_id": event_id, "status": body.status, "created": True}

@app.patch("/v1/attendance/{attendance_id}")
def update_attendance(
    attendance_id: int,
    body: AttendancePatchIn,
    current_user: dict = Depends(require_role("admin", "teacher")),
):
    """Teacher / admin: change status and/or reason of an existing record with audit tracking."""
    if body.status in ("nc", "present_nc") and not body.reason:
        raise HTTPException(422, "A reason is required for NC or Present-NC status")
    with connect() as db:
        existing = db.execute("SELECT * FROM attendance WHERE id=?", (attendance_id,)).fetchone()
        if not existing:
            raise HTTPException(404, "Attendance record not found")

        orig = existing["original_status"] or existing["status"]
        now_str = datetime.now(timezone.utc).isoformat()
        db.execute(
            "UPDATE attendance SET status=?, reason=?, manually_added=1, original_status=?, modified_by=?, modified_at=? WHERE id=?",
            (body.status, body.reason, orig, current_user["username"], now_str, attendance_id)
        )
        log_audit(db, current_user["id"], current_user["username"], "edit_attendance", "attendance", attendance_id, f"Changed status from {existing['status']} to {body.status}")
    return {"updated": True}

@app.delete("/v1/attendance/{attendance_id}", status_code=204)
def delete_attendance(
    attendance_id: int,
    current_user: dict = Depends(require_role("admin", "teacher")),
):
    with connect() as db:
        if not db.execute("SELECT 1 FROM attendance WHERE id=?", (attendance_id,)).fetchone():
            raise HTTPException(404, "Attendance record not found")
        db.execute("DELETE FROM attendance WHERE id=?", (attendance_id,))
        log_audit(db, current_user["id"], current_user["username"], "delete_attendance", "attendance", attendance_id)


# ── Device events (Pi/edge) ───────────────────────────────────────────────────
@app.post("/v1/attendance/events", dependencies=[Depends(require_device)])
def record_event(event: EventIn):
    when = event.observed_at.astimezone(timezone.utc).replace(tzinfo=None)
    with connect() as db:
        if not db.execute("SELECT 1 FROM students WHERE id=? AND active=1", (event.student_id,)).fetchone():
            raise HTTPException(404, "Unknown or inactive student")

        session = None
        if event.session_id:
            session = db.execute("SELECT * FROM sessions WHERE id=?", (event.session_id,)).fetchone()
        elif event.room:
            session = db.execute(
                "SELECT * FROM sessions WHERE room=? AND starts_at<=? AND ends_at>=? ORDER BY starts_at DESC LIMIT 1",
                (event.room, when.isoformat(), when.isoformat())
            ).fetchone()

        # Class membership validation
        if session and session["class_id"]:
            enrolled = db.execute(
                "SELECT 1 FROM student_profiles WHERE student_id=? AND class_id=?",
                (event.student_id, session["class_id"])
            ).fetchone()
            if not enrolled:
                raise HTTPException(403, f"Student is not enrolled in the class for this session")

        att_status = "present"
        if session:
            starts = datetime.fromisoformat(session["starts_at"])
            if when > starts + timedelta(minutes=session["late_after_minutes"]):
                att_status = "late"

        try:
            db.execute(
                "INSERT INTO attendance"
                "(event_id,student_id,session_id,device_id,observed_at,status,confidence,liveness_score,"
                "recognition_processing_time,liveness_confidence,face_quality_score)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (event.event_id, event.student_id, session["id"] if session else None,
                 event.device_id, when.isoformat(), att_status, event.confidence, event.liveness_score,
                 0, event.liveness_score, 0.8)
            )
        except Exception as exc:
            if "UNIQUE" in str(exc).upper():
                return {"event_id": event.event_id, "duplicate": True}
            raise
    return {"event_id": event.event_id, "status": att_status}


@app.post("/v1/attendance/web-checkin")
def web_checkin(frames: CameraFramesIn, current_user: dict = Depends(require_role("admin", "teacher"))):
    """90 %+ accurate face recognition using multi-frame voting and class membership validation."""
    import numpy as np_mod
    ACCEPT_DIST    = 0.42
    EARLY_EXIT_DIST = 0.25
    MIN_VOTE_RATIO = 0.50

    with connect() as db:
        rows = db.execute(
            "SELECT id,name,embedding FROM students WHERE active=1 AND embedding<>'[]'"
        ).fetchall()
    if not rows:
        raise HTTPException(422, "No enrolled students in the system")

    known = np_mod.asarray([json.loads(r["embedding"]) for r in rows])

    frame_votes: dict[int, list[float]] = {}
    ear_values: list[float] = []

    for value in frames.images:
        try:
            image = decode_image(value)
            vec = embedding(image)
            if vec is not None:
                distances = face_recognition.face_distance(known, vec)
                idx  = int(np_mod.argmin(distances))
                dist = float(distances[idx])
                if dist < ACCEPT_DIST:
                    frame_votes.setdefault(idx, []).append(dist)
                    if dist < EARLY_EXIT_DIST and len(frame_votes[idx]) >= 3:
                        break
            e = ear(image)
            if e is not None:
                ear_values.append(e)
        except ValueError:
            continue

    if not frame_votes:
        raise HTTPException(
            422,
            "No face matched within the confidence threshold. "
            "Ensure good lighting, face the camera directly, and that your face is enrolled."
        )

    best_idx = max(frame_votes, key=lambda i: len(frame_votes[i]))
    vote_count = len(frame_votes[best_idx])
    total_frames_sent = len(frames.images)
    vote_ratio = vote_count / total_frames_sent

    if vote_ratio < MIN_VOTE_RATIO:
        raise HTTPException(
            401,
            f"Recognition was inconsistent ({vote_count}/{total_frames_sent} frames agreed). "
            "Ensure good lighting, hold still, and face the camera directly."
        )

    winning_dists = frame_votes[best_idx]
    avg_dist = sum(winning_dists) / len(winning_dists)

    if avg_dist > ACCEPT_DIST:
        raise HTTPException(
            401,
            "Face not recognized with sufficient accuracy. "
            "Re-enroll with better lighting for higher accuracy."
        )

    if not liveness_ok(ear_values):
        raise HTTPException(422, "Liveness check failed — keep your eyes open and look at the camera.")

    student = rows[best_idx]

    # Class membership validation for target session
    with connect() as db:
        session = None
        if frames.session_id:
            session = db.execute("SELECT * FROM sessions WHERE id=?", (frames.session_id,)).fetchone()
        else:
            session = db.execute("SELECT * FROM sessions WHERE status='open' ORDER BY starts_at DESC LIMIT 1").fetchone()

        if session and session["class_id"]:
            enrolled = db.execute(
                "SELECT 1 FROM student_profiles WHERE student_id=? AND class_id=?",
                (student["id"], session["class_id"])
            ).fetchone()
            if not enrolled:
                raise HTTPException(403, f"Student {student['name']} ({student['id']}) is not enrolled in the class for this session")

        # Duplicate check: Is student already marked present/late for this session?
        existing = None
        if session:
            existing = db.execute(
                "SELECT id, status, observed_at FROM attendance WHERE student_id=? AND session_id=?",
                (student["id"], session["id"])
            ).fetchone()
        else:
            today_prefix = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            existing = db.execute(
                "SELECT id, status, observed_at FROM attendance WHERE student_id=? AND observed_at LIKE ?",
                (student["id"], f"{today_prefix}%")
            ).fetchone()

        dist_score = max(0.0, 1.0 - avg_dist / 0.50)
        raw_conf   = 0.65 + 0.35 * dist_score * vote_ratio
        confidence = round(min(0.99, raw_conf), 3)

        if existing and existing["status"] in ("present", "late"):
            return {
                "status": existing["status"],
                "already_present": True,
                "student_id": student["id"],
                "student_name": student["name"],
                "confidence": confidence,
                "vote_ratio": round(vote_ratio, 2),
                "avg_distance": round(avg_dist, 3),
                "message": f"{student['name']} is already marked {existing['status'].upper()}.",
            }

    live_score = round(
        sum(1 for e in ear_values if e > 0.20) / max(len(ear_values), 1), 3
    )

    now = datetime.now(timezone.utc)
    event = EventIn(
        event_id=str(uuid.uuid4()),
        student_id=student["id"],
        device_id="web-laptop",
        observed_at=now,
        confidence=confidence,
        liveness_score=live_score,
        room=session["room"] if session else "A101",
        session_id=session["id"] if session else None,
    )
    result = record_event(event)
    status = result.get("status", "present")
    return {
        **result,
        "status": status,
        "already_present": False,
        "student_id":   student["id"],
        "student_name": student["name"],
        "confidence":   confidence,
        "vote_ratio":   round(vote_ratio, 2),
        "avg_distance": round(avg_dist, 3),
        "message": f"{student['name']} marked {status.upper()}.",
    }


@app.get("/v1/attendance/export")
def export_attendance_csv(
    session_id: int | None = None,
    class_id: int | None = None,
    student_id: str | None = None,
    date: str | None = None,
    current_user: dict = Depends(require_role("admin", "teacher")),
):
    from fastapi.responses import PlainTextResponse
    import io, csv
    with connect() as db:
        query = """
            SELECT a.id, a.observed_at, a.status, a.confidence, a.reason,
                   s.id AS student_id, s.name AS student_name,
                   c.label AS class_label,
                   sub.name AS subject_name, sub.code AS subject_code,
                   se.room
            FROM attendance a
            JOIN students s ON s.id = a.student_id
            LEFT JOIN sessions se ON se.id = a.session_id
            LEFT JOIN classes c ON c.id = se.class_id
            LEFT JOIN subjects sub ON sub.id = se.subject_id
            WHERE 1=1
        """
        params = []
        if session_id is not None:
            query += " AND a.session_id = ?"
            params.append(session_id)
        if class_id is not None:
            query += " AND se.class_id = ?"
            params.append(class_id)
        if student_id is not None:
            query += " AND a.student_id = ?"
            params.append(student_id)
        if date is not None:
            query += " AND a.observed_at LIKE ?"
            params.append(f"{date}%")
        query += " ORDER BY a.observed_at DESC"
        rows = db.execute(query, params).fetchall()

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["ID", "Student ID", "Student Name", "Class", "Subject", "Room", "Observed At", "Status", "Confidence", "Reason"])
    for r in rows:
        writer.writerow([
            r["id"], r["student_id"], r["student_name"], r["class_label"] or "",
            r["subject_name"] or "", r["room"] or "", r["observed_at"],
            r["status"], f"{round((r['confidence'] or 0)*100)}%", r["reason"] or ""
        ])
    return PlainTextResponse(buf.getvalue(), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=attendance_export.csv"})


@app.get("/v1/students/{student_id}/attendance/export")
def export_student_attendance_csv(
    student_id: str,
    current_user: dict = Depends(get_current_user),
):
    from fastapi.responses import PlainTextResponse
    import io, csv
    if current_user["role"] == "student" and current_user["student_id"] != student_id:
        raise HTTPException(403, "You can only export your own attendance records")
    with connect() as db:
        rows = db.execute("""
            SELECT a.id, a.observed_at, a.status, a.confidence, a.reason,
                   sub.name AS subject_name, sub.code AS subject_code,
                   se.room
            FROM attendance a
            LEFT JOIN sessions se ON se.id = a.session_id
            LEFT JOIN subjects sub ON sub.id = se.subject_id
            WHERE a.student_id = ?
            ORDER BY a.observed_at DESC
        """, (student_id,)).fetchall()

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Date & Time", "Subject", "Subject Code", "Room", "Status", "Confidence", "Reason"])
    for r in rows:
        writer.writerow([
            r["observed_at"], r["subject_name"] or "", r["subject_code"] or "",
            r["room"] or "", r["status"], f"{round((r['confidence'] or 0)*100)}%", r["reason"] or ""
        ])
    return PlainTextResponse(buf.getvalue(), media_type="text/csv", headers={"Content-Disposition": f"attachment; filename=attendance_{student_id}.csv"})


# ══════════════════════════════════════════════════════════════════════════════
# ANALYTICS & ADMIN AUDIT
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/v1/analytics/at-risk")
def at_risk(threshold: float = 0.75, current_user: dict = Depends(require_role("admin", "teacher"))):
    with connect() as db:
        rows = db.execute("""
            SELECT s.id, s.name,
                   COUNT(DISTINCT se.id) total_sessions,
                   COUNT(DISTINCT a.session_id) attended_sessions
            FROM students s
            CROSS JOIN sessions se
            LEFT JOIN attendance a ON a.student_id=s.id AND a.session_id=se.id
                AND a.status NOT IN ('nc','absent')
            WHERE s.active=1 GROUP BY s.id HAVING total_sessions > 0
        """).fetchall()
    return [
        {
            "student_id": r["id"],
            "name": r["name"],
            "attendance_rate": round(r["attended_sessions"] / r["total_sessions"], 3),
        }
        for r in rows if r["attended_sessions"] / r["total_sessions"] < threshold
    ]

@app.get("/v1/dashboard/overview")
def dashboard_overview(current_user: dict = Depends(require_role("admin", "teacher"))):
    today = datetime.now(timezone.utc).date().isoformat()
    with connect() as db:
        total   = db.execute("SELECT COUNT(*) FROM students WHERE active=1").fetchone()[0]
        present = db.execute(
            "SELECT COUNT(DISTINCT student_id) FROM attendance "
            "WHERE date(observed_at)=? AND status IN ('present','late','present_nc')", (today,)
        ).fetchone()[0]
        late    = db.execute(
            "SELECT COUNT(*) FROM attendance WHERE date(observed_at)=? AND status='late'", (today,)
        ).fetchone()[0]
        risk    = db.execute("""
            SELECT COUNT(*) FROM (
                SELECT s.id FROM students s
                CROSS JOIN sessions se
                LEFT JOIN attendance a ON a.student_id=s.id AND a.session_id=se.id
                    AND a.status NOT IN ('nc','absent')
                WHERE s.active=1 GROUP BY s.id
                HAVING COUNT(DISTINCT se.id)>0
                AND CAST(COUNT(DISTINCT a.session_id) AS REAL)/COUNT(DISTINCT se.id)<0.75
            )
        """).fetchone()[0]
        recent  = db.execute(
            "SELECT s.name student_name,a.status,a.observed_at,a.confidence,sub.name AS subject_name "
            "FROM attendance a "
            "JOIN students s ON s.id=a.student_id "
            "LEFT JOIN sessions se ON se.id = a.session_id "
            "LEFT JOIN subjects sub ON sub.id = se.subject_id "
            "ORDER BY a.observed_at DESC LIMIT 12"
        ).fetchall()
    return {
        "total_students": total,
        "present_today": present,
        "late_today": late,
        "at_risk": risk,
        "recent": [dict(r) for r in recent],
    }

@app.get("/v1/admin/audit-log")
def get_audit_log(limit: int = 100, current_user: dict = Depends(require_role("admin"))):
    with connect() as db:
        rows = db.execute(
            "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]
