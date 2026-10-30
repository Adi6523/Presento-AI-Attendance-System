"""
Dedicated test suite for the NEW Presento backend foundation.

Covers all 25 required minimum tests + Critical Integration Test + Liveness Contract Tests:
 1. Database creation
 2. User creation
 3. Login
 4. Unauthorized access
 5. Student creation
 6. Student retrieval
 7. Class creation
 8. Class enrollment
 9. Duplicate class enrollment
10. Session creation
11. Session start
12. Session close
13. Valid attendance event
14. Duplicate attendance event
15. Unknown student
16. Inactive student
17. Wrong class
18. Low confidence
19. Invalid liveness (SPOOF state)
20. Closed session
21. Invalid session transition
22. Absent generation
23. Concurrent duplicate attendance
24. Dashboard summary
25. Attendance history
26. Critical end-to-end integration test
27. Health check & Swagger documentation

Liveness Contract Tests (Phase 2):
28-A. REAL_LOCKED → PRESENT
29-B. SPOOF state → BLOCKED
30-C. CHECKING state → BLOCKED
31-D. HOLD state → BLOCKED
32-E. RESET state → BLOCKED
33-F. Unknown state → BLOCKED (422 validation error)
34-G. Missing liveness_state → 422 validation error
35-H. Missing student_id → BLOCKED
36-I. Wrong class → REJECTED
37-J. Closed session → REJECTED
38-K. Concurrent dedup with REAL_LOCKED → exactly one PRESENT
39-L. AI API key enforcement (missing/wrong key → 401/403)
"""
from __future__ import annotations

import unittest
from datetime import datetime, timezone
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
import os
import tempfile
from sqlalchemy.orm import sessionmaker

from backend.database import Base, get_db
from backend.main import app
from backend.models import (
    Attendance,
    ClassModel,
    ClassStudent,
    FaceEmbedding,
    FaceProfile,
    SessionModel,
    Student,
    Teacher,
    User,
)
from backend.services.auth_service import create_access_token, hash_password


class TestPresentoBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Use an isolated temporary SQLite database file for full multi-threaded concurrency safety
        cls.temp_db_fd, cls.temp_db_path = tempfile.mkstemp(suffix="_test_presento.db")
        os.close(cls.temp_db_fd)

        cls.test_engine = create_engine(
            f"sqlite:///{cls.temp_db_path}",
            connect_args={"check_same_thread": False},
        )

        @event.listens_for(cls.test_engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        cls.TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=cls.test_engine)
        Base.metadata.create_all(bind=cls.test_engine)

        def override_get_db():
            db = cls.TestSessionLocal()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = override_get_db
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        app.dependency_overrides.clear()
        cls.test_engine.dispose()
        if os.path.exists(cls.temp_db_path):
            try:
                os.remove(cls.temp_db_path)
            except Exception:
                pass

    def setUp(self):
        # Clean data between tests
        with self.TestSessionLocal() as db:
            db.query(Attendance).delete()
            db.query(ClassStudent).delete()
            db.query(SessionModel).delete()
            db.query(ClassModel).delete()
            db.query(FaceEmbedding).delete()
            db.query(FaceProfile).delete()
            db.query(Student).delete()
            db.query(Teacher).delete()
            db.query(User).delete()
            db.commit()

    def _create_user(self, username="teacher1", password="password123", role="TEACHER"):
        with self.TestSessionLocal() as db:
            user = User(
                username=username,
                password_hash=hash_password(password),
                role=role,
                is_active=True,
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            uid = user.id
            if role == "TEACHER":
                teacher = Teacher(
                    user_id=uid,
                    name=f"Prof. {username}",
                    email=f"{username}@example.edu",
                )
                db.add(teacher)
                db.commit()
            return uid

    def _auth_headers(self, user_id: int, username="teacher1", role="TEACHER"):
        token = create_access_token({"sub": str(user_id), "username": username, "role": role})
        return {"Authorization": f"Bearer {token}"}

    def _ai_headers(self):
        """Bearer auth headers for the AI Vision engine (uses PRESENTO_AI_API_KEY)."""
        from backend.config import settings
        return {"Authorization": f"Bearer {settings.presento_ai_api_key}"}

    # ── 1. Database creation ──────────────────────────────────────────────────
    def test_01_database_creation(self):
        with self.TestSessionLocal() as db:
            result = db.execute(text("SELECT 1")).scalar()
            self.assertEqual(result, 1)
            # Verify PRAGMA foreign_keys is enabled
            fk = db.execute(text("PRAGMA foreign_keys")).scalar()
            self.assertEqual(fk, 1)

    # ── 2. User creation ──────────────────────────────────────────────────────
    def test_02_user_creation(self):
        uid = self._create_user(username="test_admin", password="secure_password", role="ADMIN")
        with self.TestSessionLocal() as db:
            user = db.query(User).filter(User.id == uid).first()
            self.assertIsNotNone(user)
            self.assertEqual(user.username, "test_admin")
            self.assertEqual(user.role, "ADMIN")
            self.assertNotEqual(user.password_hash, "secure_password")  # Hashed

    # ── 3. Login ──────────────────────────────────────────────────────────────
    def test_03_login(self):
        self._create_user(username="teacher_bob", password="mypassword123", role="TEACHER")
        resp = self.client.post("/api/auth/login", json={"username": "teacher_bob", "password": "mypassword123"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("access_token", data)
        self.assertEqual(data["token_type"], "bearer")
        self.assertEqual(data["user"]["username"], "teacher_bob")
        # Ensure password hash is NEVER exposed
        self.assertNotIn("password_hash", data["user"])

    # ── 4. Unauthorized access ────────────────────────────────────────────────
    def test_04_unauthorized_access(self):
        # Protected route without auth header
        resp = self.client.get("/api/students")
        self.assertEqual(resp.status_code, 401)
        self.assertFalse(resp.json()["success"])

        # Invalid token
        resp2 = self.client.get("/api/students", headers={"Authorization": "Bearer invalid.token.value"})
        self.assertEqual(resp2.status_code, 401)

    # ── 5. Student creation ───────────────────────────────────────────────────
    def test_05_student_creation(self):
        uid = self._create_user(username="prof1", role="TEACHER")
        headers = self._auth_headers(uid)

        payload = {
            "student_id": "23UAM090",
            "name": "Aditya Sharma",
            "email": "aditya@example.edu",
            "department": "AIML",
            "year": "4th Year",
        }
        resp = self.client.post("/api/students", json=payload, headers=headers)
        self.assertEqual(resp.status_code, 201)
        data = resp.json()
        self.assertEqual(data["student_id"], "23UAM090")
        self.assertEqual(data["name"], "Aditya Sharma")
        self.assertTrue(data["is_active"])

    # ── 6. Student retrieval ──────────────────────────────────────────────────
    def test_06_student_retrieval(self):
        uid = self._create_user(username="prof1", role="TEACHER")
        headers = self._auth_headers(uid)

        self.client.post("/api/students", json={"student_id": "23UAM091", "name": "Rahul Verma"}, headers=headers)
        self.client.post("/api/students", json={"student_id": "23UAM092", "name": "Sneha Patil"}, headers=headers)

        # Search by student_id
        resp = self.client.get("/api/students?student_id=23UAM091", headers=headers)
        self.assertEqual(resp.status_code, 200)
        results = resp.json()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["student_id"], "23UAM091")

        # Search by name
        resp2 = self.client.get("/api/students?name=Sneha", headers=headers)
        self.assertEqual(resp2.status_code, 200)
        self.assertEqual(len(resp2.json()), 1)

    # ── 7. Class creation ─────────────────────────────────────────────────────
    def test_07_class_creation(self):
        uid = self._create_user(username="prof_ai", role="TEACHER")
        headers = self._auth_headers(uid)

        payload = {
            "name": "Computer Vision",
            "subject": "CS402",
            "academic_year": "2026-2027",
            "semester": 7,
        }
        resp = self.client.post("/api/classes", json=payload, headers=headers)
        self.assertEqual(resp.status_code, 201)
        data = resp.json()
        self.assertEqual(data["name"], "Computer Vision")
        self.assertEqual(data["subject"], "CS402")

    # ── 8. Class enrollment ───────────────────────────────────────────────────
    def test_08_class_enrollment(self):
        uid = self._create_user(username="prof_ai", role="TEACHER")
        headers = self._auth_headers(uid)

        c_resp = self.client.post("/api/classes", json={"name": "ML", "subject": "CS401", "academic_year": "2026", "semester": 7}, headers=headers)
        class_id = c_resp.json()["id"]

        s_resp = self.client.post("/api/students", json={"student_id": "23UAM095", "name": "Student A"}, headers=headers)
        student_id_str = s_resp.json()["student_id"]

        enroll_resp = self.client.post(f"/api/classes/{class_id}/students/{student_id_str}", headers=headers)
        self.assertEqual(enroll_resp.status_code, 201)
        self.assertEqual(enroll_resp.json()["class_id"], class_id)

    # ── 9. Duplicate class enrollment ─────────────────────────────────────────
    def test_09_duplicate_class_enrollment(self):
        uid = self._create_user(username="prof_ai", role="TEACHER")
        headers = self._auth_headers(uid)

        c_resp = self.client.post("/api/classes", json={"name": "ML", "subject": "CS401", "academic_year": "2026", "semester": 7}, headers=headers)
        class_id = c_resp.json()["id"]
        self.client.post("/api/students", json={"student_id": "23UAM095", "name": "Student A"}, headers=headers)

        # First enrollment succeeds
        resp1 = self.client.post(f"/api/classes/{class_id}/students/23UAM095", headers=headers)
        self.assertEqual(resp1.status_code, 201)

        # Duplicate enrollment must fail with conflict (409)
        resp2 = self.client.post(f"/api/classes/{class_id}/students/23UAM095", headers=headers)
        self.assertEqual(resp2.status_code, 409)
        self.assertEqual(resp2.json()["error"], "STUDENT_ALREADY_ENROLLED")

    # ── 10. Session creation ──────────────────────────────────────────────────
    def test_10_session_creation(self):
        uid = self._create_user(username="prof_ai", role="TEACHER")
        headers = self._auth_headers(uid)

        c_resp = self.client.post("/api/classes", json={"name": "ML", "subject": "CS401", "academic_year": "2026", "semester": 7}, headers=headers)
        class_id = c_resp.json()["id"]

        sess_resp = self.client.post("/api/sessions", json={"class_id": class_id, "date": "2026-10-06"}, headers=headers)
        self.assertEqual(sess_resp.status_code, 201)
        self.assertEqual(sess_resp.json()["status"], "SCHEDULED")

    # ── 11. Session start ─────────────────────────────────────────────────────
    def test_11_session_start(self):
        uid = self._create_user(username="prof_ai", role="TEACHER")
        headers = self._auth_headers(uid)

        c_resp = self.client.post("/api/classes", json={"name": "ML", "subject": "CS401", "academic_year": "2026", "semester": 7}, headers=headers)
        class_id = c_resp.json()["id"]
        sess_resp = self.client.post("/api/sessions", json={"class_id": class_id, "date": "2026-10-06"}, headers=headers)
        session_id = sess_resp.json()["id"]

        start_resp = self.client.post(f"/api/sessions/{session_id}/start", headers=headers)
        self.assertEqual(start_resp.status_code, 200)
        self.assertEqual(start_resp.json()["status"], "ACTIVE")

        # Second session for same class cannot start concurrently
        sess_resp2 = self.client.post("/api/sessions", json={"class_id": class_id, "date": "2026-10-06"}, headers=headers)
        sess_id2 = sess_resp2.json()["id"]
        conflict_resp = self.client.post(f"/api/sessions/{sess_id2}/start", headers=headers)
        self.assertEqual(conflict_resp.status_code, 409)
        self.assertEqual(conflict_resp.json()["error"], "ACTIVE_SESSION_EXISTS")

    # ── 12. Session close ─────────────────────────────────────────────────────
    def test_12_session_close(self):
        uid = self._create_user(username="prof_ai", role="TEACHER")
        headers = self._auth_headers(uid)

        c_resp = self.client.post("/api/classes", json={"name": "ML", "subject": "CS401", "academic_year": "2026", "semester": 7}, headers=headers)
        class_id = c_resp.json()["id"]
        sess_resp = self.client.post("/api/sessions", json={"class_id": class_id, "date": "2026-10-06"}, headers=headers)
        session_id = sess_resp.json()["id"]

        self.client.post(f"/api/sessions/{session_id}/start", headers=headers)
        close_resp = self.client.post(f"/api/sessions/{session_id}/close", headers=headers)
        self.assertEqual(close_resp.status_code, 200)
        self.assertEqual(close_resp.json()["status"], "COMPLETED")

    # ── 13. Valid attendance event ────────────────────────────────────────────
    def test_13_valid_attendance_event(self):
        uid = self._create_user(username="prof_ai", role="TEACHER")
        headers = self._auth_headers(uid)

        c_resp = self.client.post("/api/classes", json={"name": "ML", "subject": "CS401", "academic_year": "2026", "semester": 7}, headers=headers)
        class_id = c_resp.json()["id"]
        self.client.post("/api/students", json={"student_id": "23UAM090", "name": "Aditya"}, headers=headers)
        self.client.post(f"/api/classes/{class_id}/students/23UAM090", headers=headers)

        sess_resp = self.client.post("/api/sessions", json={"class_id": class_id, "date": "2026-10-06"}, headers=headers)
        session_id = sess_resp.json()["id"]
        self.client.post(f"/api/sessions/{session_id}/start", headers=headers)

        event_payload = {
            "student_id": "23UAM090",
            "session_id": session_id,
            "confidence": 0.91,
            "liveness_state": "REAL_LOCKED",
            "liveness_score": 0.95,
            "source": "AI_CAMERA",
            "timestamp": "2026-10-06T10:30:00",
        }
        resp = self.client.post("/api/attendance/events", json=event_payload, headers=self._ai_headers())
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["status"], "PRESENT")

    # ── 14. Duplicate attendance event ────────────────────────────────────────
    def test_14_duplicate_attendance_event(self):
        uid = self._create_user(username="prof_ai", role="TEACHER")
        headers = self._auth_headers(uid)

        c_resp = self.client.post("/api/classes", json={"name": "ML", "subject": "CS401", "academic_year": "2026", "semester": 7}, headers=headers)
        class_id = c_resp.json()["id"]
        self.client.post("/api/students", json={"student_id": "23UAM090", "name": "Aditya"}, headers=headers)
        self.client.post(f"/api/classes/{class_id}/students/23UAM090", headers=headers)
        sess_resp = self.client.post("/api/sessions", json={"class_id": class_id, "date": "2026-10-06"}, headers=headers)
        session_id = sess_resp.json()["id"]
        self.client.post(f"/api/sessions/{session_id}/start", headers=headers)

        event_payload = {
            "student_id": "23UAM090",
            "session_id": session_id,
            "confidence": 0.91,
            "liveness_state": "REAL_LOCKED",
            "liveness_score": 0.95,
            "source": "AI_CAMERA",
        }
        # First event -> PRESENT
        resp1 = self.client.post("/api/attendance/events", json=event_payload, headers=self._ai_headers())
        self.assertEqual(resp1.json()["status"], "PRESENT")

        # Second event -> DUPLICATE
        resp2 = self.client.post("/api/attendance/events", json=event_payload, headers=self._ai_headers())
        self.assertEqual(resp2.status_code, 200)
        self.assertEqual(resp2.json()["status"], "DUPLICATE")

        # Verify exactly ONE record exists in DB
        with self.TestSessionLocal() as db:
            records = db.query(Attendance).filter(Attendance.session_id == session_id).all()
            self.assertEqual(len(records), 1)

    # ── 15. Unknown student ───────────────────────────────────────────────────
    def test_15_unknown_student(self):
        uid = self._create_user(username="prof_ai", role="TEACHER")
        headers = self._auth_headers(uid)

        c_resp = self.client.post("/api/classes", json={"name": "ML", "subject": "CS401", "academic_year": "2026", "semester": 7}, headers=headers)
        sess_resp = self.client.post("/api/sessions", json={"class_id": c_resp.json()["id"], "date": "2026-10-06"}, headers=headers)
        session_id = sess_resp.json()["id"]
        self.client.post(f"/api/sessions/{session_id}/start", headers=headers)

        resp = self.client.post(
            "/api/attendance/events",
            json={
                "student_id": "UNKNOWN_ROLL_999",
                "session_id": session_id,
                "confidence": 0.85,
                "liveness_state": "REAL_LOCKED",
            },
            headers=self._ai_headers(),
        )
        self.assertEqual(resp.status_code, 200)
        self.assertFalse(resp.json()["success"])
        self.assertEqual(resp.json()["status"], "REJECTED")

        # No row in attendance table
        with self.TestSessionLocal() as db:
            count = db.query(Attendance).count()
            self.assertEqual(count, 0)

    # ── 16. Inactive student ──────────────────────────────────────────────────
    def test_16_inactive_student(self):
        uid = self._create_user(username="prof_ai", role="TEACHER")
        headers = self._auth_headers(uid)

        c_resp = self.client.post("/api/classes", json={"name": "ML", "subject": "CS401", "academic_year": "2026", "semester": 7}, headers=headers)
        class_id = c_resp.json()["id"]
        s_resp = self.client.post("/api/students", json={"student_id": "23UAM088", "name": "Deactivated Student"}, headers=headers)
        student_pk = s_resp.json()["id"]
        self.client.post(f"/api/classes/{class_id}/students/23UAM088", headers=headers)

        # Deactivate student
        self.client.put(f"/api/students/{student_pk}", json={"is_active": False}, headers=headers)

        sess_resp = self.client.post("/api/sessions", json={"class_id": class_id, "date": "2026-10-06"}, headers=headers)
        session_id = sess_resp.json()["id"]
        self.client.post(f"/api/sessions/{session_id}/start", headers=headers)

        resp = self.client.post(
            "/api/attendance/events",
            json={
                "student_id": "23UAM088",
                "session_id": session_id,
                "confidence": 0.90,
                "liveness_state": "REAL_LOCKED",
            },
            headers=self._ai_headers(),
        )
        self.assertEqual(resp.json()["status"], "REJECTED")
        self.assertIn("inactive", resp.json()["message"].lower())

    # ── 17. Wrong class ───────────────────────────────────────────────────────
    def test_17_wrong_class(self):
        uid = self._create_user(username="prof_ai", role="TEACHER")
        headers = self._auth_headers(uid)

        c1 = self.client.post("/api/classes", json={"name": "Class A", "subject": "A1", "academic_year": "2026", "semester": 7}, headers=headers).json()["id"]
        c2 = self.client.post("/api/classes", json={"name": "Class B", "subject": "B1", "academic_year": "2026", "semester": 7}, headers=headers).json()["id"]

        self.client.post("/api/students", json={"student_id": "23UAM077", "name": "Student Class A Only"}, headers=headers)
        # Enrolled only in Class A
        self.client.post(f"/api/classes/{c1}/students/23UAM077", headers=headers)

        # Session is for Class B
        sess_b = self.client.post("/api/sessions", json={"class_id": c2, "date": "2026-10-06"}, headers=headers).json()["id"]
        self.client.post(f"/api/sessions/{sess_b}/start", headers=headers)

        # Event sent to Session B
        resp = self.client.post(
            "/api/attendance/events",
            json={
                "student_id": "23UAM077",
                "session_id": sess_b,
                "confidence": 0.92,
                "liveness_state": "REAL_LOCKED",
            },
            headers=self._ai_headers(),
        )
        self.assertEqual(resp.json()["status"], "REJECTED")
        self.assertIn("not enrolled", resp.json()["message"].lower())

    # ── 18. Low confidence ────────────────────────────────────────────────────
    def test_18_low_confidence(self):
        uid = self._create_user(username="prof_ai", role="TEACHER")
        headers = self._auth_headers(uid)

        c_resp = self.client.post("/api/classes", json={"name": "ML", "subject": "CS401", "academic_year": "2026", "semester": 7}, headers=headers)
        class_id = c_resp.json()["id"]
        self.client.post("/api/students", json={"student_id": "23UAM090", "name": "Aditya"}, headers=headers)
        self.client.post(f"/api/classes/{class_id}/students/23UAM090", headers=headers)

        sess_resp = self.client.post("/api/sessions", json={"class_id": class_id, "date": "2026-10-06"}, headers=headers)
        session_id = sess_resp.json()["id"]
        self.client.post(f"/api/sessions/{session_id}/start", headers=headers)

        # Confidence 0.35 is below 0.48 threshold
        resp = self.client.post(
            "/api/attendance/events",
            json={
                "student_id": "23UAM090",
                "session_id": session_id,
                "confidence": 0.35,
                "liveness_state": "REAL_LOCKED",
            },
            headers=self._ai_headers(),
        )
        self.assertEqual(resp.json()["status"], "BLOCKED")
        self.assertIn("threshold", resp.json()["message"].lower())

    # ── 19. Invalid liveness ──────────────────────────────────────────────────
    def test_19_invalid_liveness(self):
        uid = self._create_user(username="prof_ai", role="TEACHER")
        headers = self._auth_headers(uid)

        c_resp = self.client.post("/api/classes", json={"name": "ML", "subject": "CS401", "academic_year": "2026", "semester": 7}, headers=headers)
        class_id = c_resp.json()["id"]
        self.client.post("/api/students", json={"student_id": "23UAM090", "name": "Aditya"}, headers=headers)
        self.client.post(f"/api/classes/{class_id}/students/23UAM090", headers=headers)

        sess_resp = self.client.post("/api/sessions", json={"class_id": class_id, "date": "2026-10-06"}, headers=headers)
        session_id = sess_resp.json()["id"]
        self.client.post(f"/api/sessions/{session_id}/start", headers=headers)

        # Liveness SPOOF state — presentation attack
        resp = self.client.post(
            "/api/attendance/events",
            json={
                "student_id": "23UAM090",
                "session_id": session_id,
                "confidence": 0.95,
                "liveness_state": "SPOOF",
                "liveness_score": 0.05,
            },
            headers=self._ai_headers(),
        )
        self.assertEqual(resp.json()["status"], "BLOCKED")
        self.assertIn("spoof", resp.json()["message"].lower())

    # ── 20. Closed session ────────────────────────────────────────────────────
    def test_20_closed_session(self):
        uid = self._create_user(username="prof_ai", role="TEACHER")
        headers = self._auth_headers(uid)

        c_resp = self.client.post("/api/classes", json={"name": "ML", "subject": "CS401", "academic_year": "2026", "semester": 7}, headers=headers)
        class_id = c_resp.json()["id"]
        self.client.post("/api/students", json={"student_id": "23UAM090", "name": "Aditya"}, headers=headers)
        self.client.post(f"/api/classes/{class_id}/students/23UAM090", headers=headers)

        sess_resp = self.client.post("/api/sessions", json={"class_id": class_id, "date": "2026-10-06"}, headers=headers)
        session_id = sess_resp.json()["id"]
        self.client.post(f"/api/sessions/{session_id}/start", headers=headers)
        self.client.post(f"/api/sessions/{session_id}/close", headers=headers)

        # Attempt attendance on COMPLETED session
        resp = self.client.post(
            "/api/attendance/events",
            json={
                "student_id": "23UAM090",
                "session_id": session_id,
                "confidence": 0.95,
                "liveness_state": "REAL_LOCKED",
            },
            headers=self._ai_headers(),
        )
        self.assertEqual(resp.json()["status"], "REJECTED")
        self.assertIn("completed", resp.json()["message"].lower())

    # ── 21. Invalid session transition ────────────────────────────────────────
    def test_21_invalid_session_transition(self):
        uid = self._create_user(username="prof_ai", role="TEACHER")
        headers = self._auth_headers(uid)

        c_resp = self.client.post("/api/classes", json={"name": "ML", "subject": "CS401", "academic_year": "2026", "semester": 7}, headers=headers)
        class_id = c_resp.json()["id"]
        sess_resp = self.client.post("/api/sessions", json={"class_id": class_id, "date": "2026-10-06"}, headers=headers)
        session_id = sess_resp.json()["id"]

        # 1. Close directly from SCHEDULED (must fail)
        resp1 = self.client.post(f"/api/sessions/{session_id}/close", headers=headers)
        self.assertEqual(resp1.status_code, 400)
        self.assertEqual(resp1.json()["error"], "INVALID_STATE_TRANSITION")

        # 2. Transition SCHEDULED -> ACTIVE -> COMPLETED
        self.client.post(f"/api/sessions/{session_id}/start", headers=headers)
        self.client.post(f"/api/sessions/{session_id}/close", headers=headers)

        # 3. Transition COMPLETED -> ACTIVE (must fail)
        resp2 = self.client.post(f"/api/sessions/{session_id}/start", headers=headers)
        self.assertEqual(resp2.status_code, 400)
        self.assertEqual(resp2.json()["error"], "INVALID_STATE_TRANSITION")

        # 4. Transition COMPLETED -> CANCELLED (must fail)
        resp3 = self.client.post(f"/api/sessions/{session_id}/cancel", headers=headers)
        self.assertEqual(resp3.status_code, 400)
        self.assertEqual(resp3.json()["error"], "INVALID_STATE_TRANSITION")

    # ── 22. Absent generation ─────────────────────────────────────────────────
    def test_22_absent_generation(self):
        uid = self._create_user(username="prof_ai", role="TEACHER")
        headers = self._auth_headers(uid)

        c_resp = self.client.post("/api/classes", json={"name": "ML", "subject": "CS401", "academic_year": "2026", "semester": 7}, headers=headers)
        class_id = c_resp.json()["id"]

        # Enroll 2 students: Aditya and Rahul
        self.client.post("/api/students", json={"student_id": "23UAM090", "name": "Aditya"}, headers=headers)
        self.client.post("/api/students", json={"student_id": "23UAM091", "name": "Rahul"}, headers=headers)
        self.client.post(f"/api/classes/{class_id}/students/23UAM090", headers=headers)
        self.client.post(f"/api/classes/{class_id}/students/23UAM091", headers=headers)

        sess_resp = self.client.post("/api/sessions", json={"class_id": class_id, "date": "2026-10-06"}, headers=headers)
        session_id = sess_resp.json()["id"]
        self.client.post(f"/api/sessions/{session_id}/start", headers=headers)

        # Only Aditya attends
        self.client.post(
            "/api/attendance/events",
            json={
                "student_id": "23UAM090",
                "session_id": session_id,
                "confidence": 0.90,
                "liveness_state": "REAL_LOCKED",
            },
            headers=self._ai_headers(),
        )

        # Close session
        self.client.post(f"/api/sessions/{session_id}/close", headers=headers)

        # Query session attendance: Aditya should be PRESENT, Rahul should be ABSENT
        att_resp = self.client.get(f"/api/sessions/{session_id}/attendance", headers=headers)
        self.assertEqual(att_resp.status_code, 200)
        records = att_resp.json()
        self.assertEqual(len(records), 2)

        status_by_student = {r["student_id"]: r["status"] for r in records}
        self.assertEqual(status_by_student["23UAM090"], "PRESENT")
        self.assertEqual(status_by_student["23UAM091"], "ABSENT")

    # ── 23. Concurrent duplicate attendance ───────────────────────────────────
    def test_23_concurrent_duplicate_attendance(self):
        uid = self._create_user(username="prof_ai", role="TEACHER")
        headers = self._auth_headers(uid)

        c_resp = self.client.post("/api/classes", json={"name": "ML", "subject": "CS401", "academic_year": "2026", "semester": 7}, headers=headers)
        class_id = c_resp.json()["id"]
        self.client.post("/api/students", json={"student_id": "23UAM090", "name": "Aditya"}, headers=headers)
        self.client.post(f"/api/classes/{class_id}/students/23UAM090", headers=headers)

        sess_resp = self.client.post("/api/sessions", json={"class_id": class_id, "date": "2026-10-06"}, headers=headers)
        session_id = sess_resp.json()["id"]
        self.client.post(f"/api/sessions/{session_id}/start", headers=headers)

        import concurrent.futures

        def post_simultaneous():
            return self.client.post(
                "/api/attendance/events",
                json={
                    "student_id": "23UAM090",
                    "session_id": session_id,
                    "confidence": 0.92,
                    "liveness_state": "REAL_LOCKED",
                },
                headers=self._ai_headers(),
            )

        # Simulate 5 concurrent threads posting attendance simultaneously
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(post_simultaneous) for _ in range(5)]
            responses = [f.result() for f in futures]

        # All requests must succeed without unhandled exceptions
        statuses = [r.json()["status"] for r in responses]
        self.assertIn("PRESENT", statuses)
        self.assertEqual(statuses.count("PRESENT"), 1)
        self.assertEqual(statuses.count("DUPLICATE"), 4)

        # Database must strictly contain exactly 1 attendance record
        with self.TestSessionLocal() as db:
            count = db.query(Attendance).filter(Attendance.session_id == session_id).count()
            self.assertEqual(count, 1)

    # ── 24. Dashboard summary ─────────────────────────────────────────────────
    def test_24_dashboard_summary(self):
        uid = self._create_user(username="prof_ai", role="TEACHER")
        headers = self._auth_headers(uid)

        today_str = datetime.utcnow().strftime("%Y-%m-%d")

        c_resp = self.client.post("/api/classes", json={"name": "ML", "subject": "CS401", "academic_year": "2026", "semester": 7}, headers=headers)
        class_id = c_resp.json()["id"]
        self.client.post("/api/students", json={"student_id": "23UAM090", "name": "Aditya"}, headers=headers)
        self.client.post(f"/api/classes/{class_id}/students/23UAM090", headers=headers)

        sess_resp = self.client.post("/api/sessions", json={"class_id": class_id, "date": today_str}, headers=headers)
        session_id = sess_resp.json()["id"]
        self.client.post(f"/api/sessions/{session_id}/start", headers=headers)
        self.client.post(
            "/api/attendance/events",
            json={
                "student_id": "23UAM090",
                "session_id": session_id,
                "confidence": 0.90,
                "liveness_state": "REAL_LOCKED",
            },
            headers=self._ai_headers(),
        )



        resp = self.client.get("/api/dashboard/summary", headers=headers)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["total_students"], 1)
        self.assertEqual(data["total_classes"], 1)
        self.assertEqual(data["today_sessions"], 1)
        self.assertEqual(data["present_today"], 1)
        self.assertEqual(data["attendance_percentage"], 100.0)

    # ── 25. Attendance history ────────────────────────────────────────────────
    def test_25_attendance_history(self):
        uid = self._create_user(username="prof_ai", role="TEACHER")
        headers = self._auth_headers(uid)

        c_resp = self.client.post("/api/classes", json={"name": "ML", "subject": "CS401", "academic_year": "2026", "semester": 7}, headers=headers)
        class_id = c_resp.json()["id"]
        self.client.post("/api/students", json={"student_id": "23UAM090", "name": "Aditya"}, headers=headers)
        self.client.post(f"/api/classes/{class_id}/students/23UAM090", headers=headers)

        sess_resp = self.client.post("/api/sessions", json={"class_id": class_id, "date": "2026-10-06"}, headers=headers)
        session_id = sess_resp.json()["id"]
        self.client.post(f"/api/sessions/{session_id}/start", headers=headers)
        self.client.post(
            "/api/attendance/events",
            json={
                "student_id": "23UAM090",
                "session_id": session_id,
                "confidence": 0.90,
                "liveness_state": "REAL_LOCKED",
            },
            headers=self._ai_headers(),
        )

        hist_resp = self.client.get("/api/attendance/history?status=PRESENT", headers=headers)
        self.assertEqual(hist_resp.status_code, 200)
        records = hist_resp.json()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["student_id"], "23UAM090")
        self.assertEqual(records[0]["status"], "PRESENT")

    # ── 26. Critical End-to-End Integration Test ─────────────────────────────
    def test_26_critical_e2e_integration_flow(self):
        """
        End-to-End Pipeline:
        Teacher login
            ↓
        Create student
            ↓
        Create class
            ↓
        Enroll student
            ↓
        Create session
            ↓
        Start session
            ↓
        Send valid AI attendance event
            ↓
        PRESENT
            ↓
        Query database
            ↓
        Exactly one attendance record
        Then send the same event again:
        Expected: DUPLICATE
        Database must still contain: ONE PRESENT record.
        """
        # 1. Create Teacher and Login
        self._create_user(username="prof_demo", password="secretpassword", role="TEACHER")
        login_resp = self.client.post("/api/auth/login", json={"username": "prof_demo", "password": "secretpassword"})
        self.assertEqual(login_resp.status_code, 200)
        token = login_resp.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # 2. Create student
        stu_resp = self.client.post(
            "/api/students",
            json={
                "student_id": "23UAM099",
                "name": "Integration Student",
                "email": "student99@example.edu",
                "department": "AIML",
                "year": "4th Year",
            },
            headers=headers,
        )
        self.assertEqual(stu_resp.status_code, 201)
        student_id_str = stu_resp.json()["student_id"]

        # 3. Create class
        cls_resp = self.client.post(
            "/api/classes",
            json={
                "name": "E2E Distributed Systems",
                "subject": "CS500",
                "academic_year": "2026-2027",
                "semester": 8,
            },
            headers=headers,
        )
        self.assertEqual(cls_resp.status_code, 201)
        class_id = cls_resp.json()["id"]

        # 4. Enroll student
        enr_resp = self.client.post(f"/api/classes/{class_id}/students/{student_id_str}", headers=headers)
        self.assertEqual(enr_resp.status_code, 201)

        # 5. Create session
        sess_resp = self.client.post(
            "/api/sessions",
            json={"class_id": class_id, "date": "2026-10-06"},
            headers=headers,
        )
        self.assertEqual(sess_resp.status_code, 201)
        session_id = sess_resp.json()["id"]

        # 6. Start session
        start_resp = self.client.post(f"/api/sessions/{session_id}/start", headers=headers)
        self.assertEqual(start_resp.status_code, 200)
        self.assertEqual(start_resp.json()["status"], "ACTIVE")

        # 7. Send valid AI attendance event
        ai_event = {
            "student_id": student_id_str,
            "session_id": session_id,
            "confidence": 0.94,
            "liveness_state": "REAL_LOCKED",
            "liveness_score": 0.98,
            "source": "AI_CAMERA",
            "timestamp": "2026-10-06T10:45:00",
        }
        event_resp1 = self.client.post("/api/attendance/events", json=ai_event, headers=self._ai_headers())
        self.assertEqual(event_resp1.status_code, 200)
        self.assertEqual(event_resp1.json()["status"], "PRESENT")

        # 8. Query database -> exactly one attendance record
        with self.TestSessionLocal() as db:
            records = db.query(Attendance).filter(Attendance.session_id == session_id).all()
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].status, "PRESENT")
            self.assertAlmostEqual(records[0].confidence, 0.94, places=2)

        # 9. Send same event again
        event_resp2 = self.client.post("/api/attendance/events", json=ai_event, headers=self._ai_headers())
        self.assertEqual(event_resp2.status_code, 200)
        self.assertEqual(event_resp2.json()["status"], "DUPLICATE")

        # 10. Database must still contain ONE PRESENT record
        with self.TestSessionLocal() as db:
            records_after = db.query(Attendance).filter(Attendance.session_id == session_id).all()
            self.assertEqual(len(records_after), 1)
            self.assertEqual(records_after[0].status, "PRESENT")

    # ── 27. Health Check & Swagger Documentation ──────────────────────────────
    def test_27_health_and_docs(self):
        # Health check
        health_resp = self.client.get("/health")
        self.assertEqual(health_resp.status_code, 200)
        self.assertEqual(health_resp.json()["status"], "ok")
        self.assertEqual(health_resp.json()["service"], "presento-backend")
        self.assertEqual(health_resp.json()["database"], "connected")

        # Swagger docs
        docs_resp = self.client.get("/docs")
        self.assertEqual(docs_resp.status_code, 200)

        # OpenAPI JSON schema
        openapi_resp = self.client.get("/openapi.json")
        self.assertEqual(openapi_resp.status_code, 200)
        schema = openapi_resp.json()
        self.assertIn("/api/attendance/events", schema["paths"])
        self.assertIn("/api/auth/login", schema["paths"])

    # ════════════════════════════════════════════════════════════════════════════
    # LIVENESS CONTRACT TESTS (Phase 2 — Tests A–L)
    # ════════════════════════════════════════════════════════════════════════════

    def _make_active_session(self, headers):
        """Helper: create class, enroll student 23UAM090, start session. Returns session_id."""
        c_resp = self.client.post(
            "/api/classes",
            json={"name": "CV Class", "subject": "CS402", "academic_year": "2026", "semester": 7},
            headers=headers,
        )
        class_id = c_resp.json()["id"]
        self.client.post("/api/students", json={"student_id": "23UAM090", "name": "Aditya"}, headers=headers)
        self.client.post(f"/api/classes/{class_id}/students/23UAM090", headers=headers)
        sess_resp = self.client.post("/api/sessions", json={"class_id": class_id, "date": "2026-10-07"}, headers=headers)
        session_id = sess_resp.json()["id"]
        self.client.post(f"/api/sessions/{session_id}/start", headers=headers)
        return session_id, class_id

    # ── Test A: REAL_LOCKED → PRESENT ─────────────────────────────────────────
    def test_28A_liveness_real_locked_admitted(self):
        """liveness_state=REAL_LOCKED with valid confidence → PRESENT."""
        uid = self._create_user(username="prof_a", role="TEACHER")
        headers = self._auth_headers(uid, username="prof_a")
        session_id, _ = self._make_active_session(headers)

        resp = self.client.post(
            "/api/attendance/events",
            json={"student_id": "23UAM090", "session_id": session_id, "confidence": 0.92, "liveness_state": "REAL_LOCKED"},
            headers=self._ai_headers(),
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "PRESENT")
        self.assertTrue(resp.json()["success"])

    # ── Test B: SPOOF state → BLOCKED ─────────────────────────────────────────
    def test_29B_liveness_spoof_blocked(self):
        """liveness_state=SPOOF → BLOCKED with 'spoof' in message."""
        uid = self._create_user(username="prof_b", role="TEACHER")
        headers = self._auth_headers(uid, username="prof_b")
        session_id, _ = self._make_active_session(headers)

        resp = self.client.post(
            "/api/attendance/events",
            json={"student_id": "23UAM090", "session_id": session_id, "confidence": 0.92,
                  "liveness_state": "SPOOF", "liveness_score": 0.04},
            headers=self._ai_headers(),
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "BLOCKED")
        self.assertFalse(resp.json()["success"])
        self.assertIn("spoof", resp.json()["message"].lower())

        # No attendance row
        with self.TestSessionLocal() as db:
            self.assertEqual(db.query(Attendance).count(), 0)

    # ── Test C: CHECKING state → BLOCKED ──────────────────────────────────────
    def test_30C_liveness_checking_blocked(self):
        """liveness_state=CHECKING (verification in progress) → BLOCKED."""
        uid = self._create_user(username="prof_c", role="TEACHER")
        headers = self._auth_headers(uid, username="prof_c")
        session_id, _ = self._make_active_session(headers)

        resp = self.client.post(
            "/api/attendance/events",
            json={"student_id": "23UAM090", "session_id": session_id, "confidence": 0.88, "liveness_state": "CHECKING"},
            headers=self._ai_headers(),
        )
        self.assertEqual(resp.json()["status"], "BLOCKED")
        self.assertIn("incomplete", resp.json()["message"].lower())

    # ── Test D: HOLD state → BLOCKED ──────────────────────────────────────────
    def test_31D_liveness_hold_blocked(self):
        """liveness_state=HOLD (lease about to expire) → BLOCKED."""
        uid = self._create_user(username="prof_d", role="TEACHER")
        headers = self._auth_headers(uid, username="prof_d")
        session_id, _ = self._make_active_session(headers)

        resp = self.client.post(
            "/api/attendance/events",
            json={"student_id": "23UAM090", "session_id": session_id, "confidence": 0.88, "liveness_state": "HOLD"},
            headers=self._ai_headers(),
        )
        self.assertEqual(resp.json()["status"], "BLOCKED")
        self.assertIn("incomplete", resp.json()["message"].lower())

    # ── Test E: RESET state → BLOCKED ─────────────────────────────────────────
    def test_32E_liveness_reset_blocked(self):
        """liveness_state=RESET → BLOCKED."""
        uid = self._create_user(username="prof_e", role="TEACHER")
        headers = self._auth_headers(uid, username="prof_e")
        session_id, _ = self._make_active_session(headers)

        resp = self.client.post(
            "/api/attendance/events",
            json={"student_id": "23UAM090", "session_id": session_id, "confidence": 0.88, "liveness_state": "RESET"},
            headers=self._ai_headers(),
        )
        self.assertEqual(resp.json()["status"], "BLOCKED")

    # ── Test F: Unknown liveness_state → 422 Validation Error ─────────────────
    def test_33F_liveness_unknown_state_validation_error(self):
        """liveness_state with an unrecognized string → 422 Unprocessable Entity."""
        uid = self._create_user(username="prof_f", role="TEACHER")
        headers = self._auth_headers(uid, username="prof_f")
        session_id, _ = self._make_active_session(headers)

        resp = self.client.post(
            "/api/attendance/events",
            json={"student_id": "23UAM090", "session_id": session_id, "confidence": 0.88,
                  "liveness_state": "TOTALLY_FAKE_STATE"},
            headers=self._ai_headers(),
        )
        self.assertEqual(resp.status_code, 422)

    # ── Test G: Missing liveness_state → 422 Validation Error ─────────────────
    def test_34G_missing_liveness_state_validation_error(self):
        """liveness_state is a required field; omitting it → 422."""
        uid = self._create_user(username="prof_g", role="TEACHER")
        headers = self._auth_headers(uid, username="prof_g")
        session_id, _ = self._make_active_session(headers)

        resp = self.client.post(
            "/api/attendance/events",
            # No liveness_state at all
            json={"student_id": "23UAM090", "session_id": session_id, "confidence": 0.88},
            headers=self._ai_headers(),
        )
        self.assertEqual(resp.status_code, 422)

    # ── Test H: Missing student_id (unknown face) → BLOCKED ───────────────────
    def test_35H_missing_student_id_blocked(self):
        """No student_id (null) with REAL_LOCKED → BLOCKED (unknown face)."""
        uid = self._create_user(username="prof_h", role="TEACHER")
        headers = self._auth_headers(uid, username="prof_h")
        session_id, _ = self._make_active_session(headers)

        resp = self.client.post(
            "/api/attendance/events",
            json={"student_id": None, "session_id": session_id, "confidence": 0.92, "liveness_state": "REAL_LOCKED"},
            headers=self._ai_headers(),
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "BLOCKED")
        self.assertIn("unknown", resp.json()["message"].lower())

    # ── Test I: Wrong class → REJECTED ────────────────────────────────────────
    def test_36I_wrong_class_rejected(self):
        """REAL_LOCKED but student enrolled in different class → REJECTED."""
        uid = self._create_user(username="prof_i", role="TEACHER")
        headers = self._auth_headers(uid, username="prof_i")

        c1 = self.client.post("/api/classes", json={"name": "Class X", "subject": "X1", "academic_year": "2026", "semester": 7}, headers=headers).json()["id"]
        c2 = self.client.post("/api/classes", json={"name": "Class Y", "subject": "Y1", "academic_year": "2026", "semester": 7}, headers=headers).json()["id"]
        self.client.post("/api/students", json={"student_id": "23UAM090", "name": "Aditya"}, headers=headers)
        self.client.post(f"/api/classes/{c1}/students/23UAM090", headers=headers)  # Enrolled in c1 only
        sess = self.client.post("/api/sessions", json={"class_id": c2, "date": "2026-10-07"}, headers=headers).json()["id"]
        self.client.post(f"/api/sessions/{sess}/start", headers=headers)

        resp = self.client.post(
            "/api/attendance/events",
            json={"student_id": "23UAM090", "session_id": sess, "confidence": 0.92, "liveness_state": "REAL_LOCKED"},
            headers=self._ai_headers(),
        )
        self.assertEqual(resp.json()["status"], "REJECTED")
        self.assertIn("not enrolled", resp.json()["message"].lower())

    # ── Test J: Closed session → REJECTED ─────────────────────────────────────
    def test_37J_closed_session_rejected(self):
        """REAL_LOCKED but session is COMPLETED → REJECTED."""
        uid = self._create_user(username="prof_j", role="TEACHER")
        headers = self._auth_headers(uid, username="prof_j")
        session_id, _ = self._make_active_session(headers)
        self.client.post(f"/api/sessions/{session_id}/close", headers=headers)

        resp = self.client.post(
            "/api/attendance/events",
            json={"student_id": "23UAM090", "session_id": session_id, "confidence": 0.92, "liveness_state": "REAL_LOCKED"},
            headers=self._ai_headers(),
        )
        self.assertEqual(resp.json()["status"], "REJECTED")
        self.assertIn("completed", resp.json()["message"].lower())

    # ── Test K: Concurrent dedup with REAL_LOCKED ─────────────────────────────
    def test_38K_concurrent_real_locked_dedup(self):
        """5 concurrent REAL_LOCKED events → exactly 1 PRESENT, 4 DUPLICATE."""
        uid = self._create_user(username="prof_k", role="TEACHER")
        headers = self._auth_headers(uid, username="prof_k")
        session_id, _ = self._make_active_session(headers)

        import concurrent.futures

        def post():
            return self.client.post(
                "/api/attendance/events",
                json={"student_id": "23UAM090", "session_id": session_id, "confidence": 0.92, "liveness_state": "REAL_LOCKED"},
                headers=self._ai_headers(),
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
            results = [f.result() for f in [ex.submit(post) for _ in range(5)]]

        statuses = [r.json()["status"] for r in results]
        self.assertEqual(statuses.count("PRESENT"), 1)
        self.assertEqual(statuses.count("DUPLICATE"), 4)

        with self.TestSessionLocal() as db:
            self.assertEqual(db.query(Attendance).filter(Attendance.session_id == session_id).count(), 1)

    # ── Test L: AI API key enforcement ────────────────────────────────────────
    def test_39L_ai_api_key_enforcement(self):
        """POST /api/attendance/events requires valid PRESENTO_AI_API_KEY."""
        uid = self._create_user(username="prof_l", role="TEACHER")
        headers = self._auth_headers(uid, username="prof_l")
        session_id, _ = self._make_active_session(headers)

        payload = {"student_id": "23UAM090", "session_id": session_id, "confidence": 0.92, "liveness_state": "REAL_LOCKED"}

        # 1. No Authorization header → 401
        resp_no_key = self.client.post("/api/attendance/events", json=payload)
        self.assertEqual(resp_no_key.status_code, 401)

        # 2. Wrong key → 403
        resp_wrong_key = self.client.post(
            "/api/attendance/events", json=payload,
            headers={"Authorization": "Bearer totally-wrong-key-12345"},
        )
        self.assertEqual(resp_wrong_key.status_code, 403)

        # 3. Frontend JWT (valid user token) must NOT unlock the AI endpoint → 403
        resp_jwt = self.client.post(
            "/api/attendance/events", json=payload,
            headers=self._auth_headers(uid, username="prof_l"),  # Frontend JWT, not AI key
        )
        self.assertEqual(resp_jwt.status_code, 403)

        # 4. Correct AI key → 200 PRESENT
        resp_ok = self.client.post("/api/attendance/events", json=payload, headers=self._ai_headers())
        self.assertEqual(resp_ok.status_code, 200)
        self.assertEqual(resp_ok.json()["status"], "PRESENT")


if __name__ == "__main__":
    unittest.main()
