import os
import tempfile
import json
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

# Configure test environment before imports
os.environ["API_TOKEN"] = "test-token-12345"
os.environ["JWT_SECRET_KEY"] = "test-jwt-secret"

from backend.main import app
from backend import database
from backend.auth import hash_password, verify_password, create_access_token
from edge import api


def _jwt_headers(role="teacher", username="teacher1", student_id=None):
    """Helper: create a JWT bearer token for a given role."""
    token = create_access_token({"sub": "1", "role": role})
    return {"Authorization": f"Bearer {token}"}


def _seed_user(db, uid, username, password, role, student_id=None):
    db.execute(
        "INSERT OR IGNORE INTO users(id,username,email,password_hash,role,student_id) VALUES(?,?,NULL,?,?,?)",
        (uid, username, hash_password(password), role, student_id),
    )


# ══════════════════════════════════════════════════════════════════════════════
class TestHealth(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        database.DATA_DIR = Path(self.tmp.name)
        database.DB_PATH = database.DATA_DIR / "presento.db"
        database.initialize()
        self.client = TestClient(app)

    def tearDown(self):
        self.tmp.cleanup()

    def test_health(self):
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json(), {"ok": True})


# ══════════════════════════════════════════════════════════════════════════════
class TestAuth(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        database.DATA_DIR = Path(self.tmp.name)
        database.DB_PATH = database.DATA_DIR / "presento.db"
        database.initialize()
        self.client = TestClient(app)
        # Seed users
        from backend.database import connect
        with connect() as db:
            _seed_user(db, 1, "admin1",   "admin123",   "admin")
            _seed_user(db, 2, "teacher1", "teacher123", "teacher")
            _seed_user(db, 3, "23UAM090", "student123", "student", "23UAM090")

    def tearDown(self):
        self.tmp.cleanup()

    def test_login_success_teacher(self):
        res = self.client.post("/auth/login", data={"username": "teacher1", "password": "teacher123"})
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertIn("access_token", body)
        self.assertEqual(body["role"], "teacher")

    def test_login_success_student(self):
        res = self.client.post("/auth/login", data={"username": "23UAM090", "password": "student123"})
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(body["role"], "student")
        self.assertEqual(body["student_id"], "23UAM090")

    def test_login_wrong_password(self):
        res = self.client.post("/auth/login", data={"username": "teacher1", "password": "wrongpass"})
        self.assertEqual(res.status_code, 401)

    def test_login_unknown_user(self):
        res = self.client.post("/auth/login", data={"username": "ghost", "password": "x"})
        self.assertEqual(res.status_code, 401)

    def test_me_endpoint(self):
        # Login, then call /auth/me
        res = self.client.post("/auth/login", data={"username": "teacher1", "password": "teacher123"})
        token = res.json()["access_token"]
        me = self.client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(me.status_code, 200)
        self.assertEqual(me.json()["username"], "teacher1")
        self.assertNotIn("password_hash", me.json())

    def test_admin_can_create_user(self):
        res = self.client.post("/auth/login", data={"username": "admin1", "password": "admin123"})
        token = res.json()["access_token"]
        create = self.client.post(
            "/auth/register",
            headers={"Authorization": f"Bearer {token}"},
            json={"username": "newteacher", "password": "pass123", "role": "teacher"},
        )
        self.assertEqual(create.status_code, 201)
        self.assertTrue(create.json()["created"])

    def test_teacher_cannot_create_admin(self):
        res = self.client.post("/auth/login", data={"username": "teacher1", "password": "teacher123"})
        token = res.json()["access_token"]
        create = self.client.post(
            "/auth/register",
            headers={"Authorization": f"Bearer {token}"},
            json={"username": "badmin", "password": "pass123", "role": "admin"},
        )
        self.assertEqual(create.status_code, 403)

    def test_student_cannot_access_teacher_route(self):
        res = self.client.post("/auth/login", data={"username": "23UAM090", "password": "student123"})
        token = res.json()["access_token"]
        route = self.client.get("/v1/students", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(route.status_code, 403)

    def test_no_token_returns_401(self):
        res = self.client.get("/v1/students")
        self.assertEqual(res.status_code, 401)

    def test_bcrypt_hash_verify(self):
        h = hash_password("mysecret")
        self.assertTrue(verify_password("mysecret", h))
        self.assertFalse(verify_password("wrongpass", h))


# ══════════════════════════════════════════════════════════════════════════════
class TestStudentDataIsolation(unittest.TestCase):
    """Student role can only see their own attendance, not another student's."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        database.DATA_DIR = Path(self.tmp.name)
        database.DB_PATH = database.DATA_DIR / "presento.db"
        database.initialize()
        self.client = TestClient(app)
        dh = {"X-API-Token": "test-token-12345"}  # device header for edge endpoints
        from backend.database import connect
        # Create students
        self.client.post("/v1/students", headers=dh, json={"id": "S1", "name": "Alice", "embedding": [0.1]*128})
        self.client.post("/v1/students", headers=dh, json={"id": "S2", "name": "Bob",   "embedding": [0.2]*128})
        # Create user accounts
        with connect() as db:
            _seed_user(db, 10, "alice", "alice123", "student", "S1")
            _seed_user(db, 11, "bob",   "bob123",   "student", "S2")
            _seed_user(db, 12, "teach", "teach123", "teacher")

    def tearDown(self):
        self.tmp.cleanup()

    def _login_token(self, username, password):
        res = self.client.post("/auth/login", data={"username": username, "password": password})
        return res.json()["access_token"]

    def test_student_can_view_own_attendance(self):
        tok = self._login_token("alice", "alice123")
        res = self.client.get("/v1/students/S1/attendance", headers={"Authorization": f"Bearer {tok}"})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["student"]["id"], "S1")

    def test_student_cannot_view_other_student_attendance(self):
        tok = self._login_token("alice", "alice123")
        res = self.client.get("/v1/students/S2/attendance", headers={"Authorization": f"Bearer {tok}"})
        self.assertEqual(res.status_code, 403)

    def test_teacher_can_view_any_student_attendance(self):
        tok = self._login_token("teach", "teach123")
        for sid in ("S1", "S2"):
            res = self.client.get(f"/v1/students/{sid}/attendance", headers={"Authorization": f"Bearer {tok}"})
            self.assertEqual(res.status_code, 200)


# ══════════════════════════════════════════════════════════════════════════════
class TestBackend(unittest.TestCase):
    """Core API functionality tests — sessions, attendance rules, analytics."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        database.DATA_DIR = Path(self.tmp.name)
        database.DB_PATH = database.DATA_DIR / "presento.db"
        database.initialize()
        self.client = TestClient(app)
        self.dh = {"X-API-Token": "test-token-12345"}   # device (edge) token
        # Create a teacher JWT and admin JWT
        from backend.database import connect
        with connect() as db:
            _seed_user(db, 1, "teacher1", "pass", "teacher")
            _seed_user(db, 2, "admin1",   "pass", "admin")
        tok = self.client.post("/auth/login", data={"username": "teacher1", "password": "pass"}).json()["access_token"]
        self.jh = {"Authorization": f"Bearer {tok}"}    # Teacher JWT header
        admin_tok = self.client.post("/auth/login", data={"username": "admin1", "password": "pass"}).json()["access_token"]
        self.admin_jh = {"Authorization": f"Bearer {admin_tok}"}

    def tearDown(self):
        self.tmp.cleanup()

    def test_auth_rejection_no_token(self):
        res = self.client.get("/v1/students")
        self.assertEqual(res.status_code, 401)

    def test_device_auth_rejected_on_jwt_route(self):
        # Device token not accepted on JWT-protected routes
        res = self.client.get("/v1/students", headers=self.dh)
        self.assertEqual(res.status_code, 401)

    def test_student_lifecycle(self):
        # Teacher cannot create student manually (403)
        res_teacher = self.client.post(
            "/v1/students/manual", headers=self.jh,
            json={"id": "23UAM001", "name": "Aditya Test", "email": "a@test.com"}
        )
        self.assertEqual(res_teacher.status_code, 403)

        # Admin creates student manual record (201)
        res = self.client.post(
            "/v1/students/manual", headers=self.admin_jh,
            json={"id": "23UAM001", "name": "Aditya Test", "email": "a@test.com"}
        )
        self.assertEqual(res.status_code, 201)
        self.assertTrue(res.json().get("enrollment_required"))

        # List students (JWT)
        res = self.client.get("/v1/students", headers=self.admin_jh)
        students = res.json()
        self.assertEqual(len(students), 1)
        self.assertFalse(students[0]["enrolled"])

        # Create student with embedding (device token — edge path)
        res = self.client.post(
            "/v1/students", headers=self.dh,
            json={"id": "23UAM002", "name": "Bob", "embedding": [0.05]*128}
        )
        self.assertEqual(res.status_code, 201)

        # Embeddings endpoint uses device token
        res = self.client.get("/v1/students/embeddings", headers=self.dh)
        data = res.json()
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["id"], "23UAM002")
        self.assertEqual(len(data[0]["embedding"]), 128)

    def test_sessions_and_attendance_rules(self):
        self.client.post("/v1/students", headers=self.dh, json={"id": "S10", "name": "Charlie", "embedding": [0.1]*128})
        now = datetime.now(timezone.utc).replace(microsecond=0)
        start = now - timedelta(minutes=5)
        end   = now + timedelta(minutes=55)

        res = self.client.post("/v1/sessions", headers=self.jh, json={
            "course_code": "CS-101", "room": "A101",
            "starts_at": start.isoformat(), "ends_at": end.isoformat(), "late_after_minutes": 10,
        })
        self.assertEqual(res.status_code, 201)

        # On-time
        res = self.client.post("/v1/attendance/events", headers=self.dh, json={
            "event_id": "evt-001", "student_id": "S10", "device_id": "pi",
            "observed_at": now.isoformat(), "confidence": 0.95, "liveness_score": 1.0, "room": "A101",
        })
        self.assertEqual(res.json()["status"], "present")

        # Duplicate
        res = self.client.post("/v1/attendance/events", headers=self.dh, json={
            "event_id": "evt-001", "student_id": "S10", "device_id": "pi",
            "observed_at": now.isoformat(), "confidence": 0.95, "liveness_score": 1.0, "room": "A101",
        })
        self.assertTrue(res.json().get("duplicate"))

        # Late
        late = start + timedelta(minutes=15)
        res = self.client.post("/v1/attendance/events", headers=self.dh, json={
            "event_id": "evt-002", "student_id": "S10", "device_id": "pi",
            "observed_at": late.isoformat(), "confidence": 0.92, "liveness_score": 1.0, "room": "A101",
        })
        self.assertEqual(res.json()["status"], "late")

    def test_analytics_and_dashboard(self):
        self.client.post("/v1/students", headers=self.dh, json={"id": "S1", "name": "Good", "embedding": [0.2]*128})
        self.client.post("/v1/students", headers=self.dh, json={"id": "S2", "name": "Risk", "embedding": [0.3]*128})
        now = datetime.now(timezone.utc).replace(microsecond=0)
        self.client.post("/v1/sessions", headers=self.jh, json={
            "course_code": "CS-102", "room": "B202",
            "starts_at": (now - timedelta(hours=2)).isoformat(),
            "ends_at":   (now - timedelta(hours=1)).isoformat(), "late_after_minutes": 10,
        })
        self.client.post("/v1/attendance/events", headers=self.dh, json={
            "event_id": "evt-good", "student_id": "S1", "device_id": "pi",
            "observed_at": (now - timedelta(minutes=115)).isoformat(),
            "confidence": 0.98, "liveness_score": 1.0, "room": "B202",
        })

        risk = self.client.get("/v1/analytics/at-risk?threshold=0.75", headers=self.jh).json()
        self.assertEqual(len(risk), 1)
        self.assertEqual(risk[0]["student_id"], "S2")

        overview = self.client.get("/v1/dashboard/overview", headers=self.jh).json()
        self.assertEqual(overview["total_students"], 2)
        self.assertEqual(overview["at_risk"], 1)


# ══════════════════════════════════════════════════════════════════════════════
class TestEdge(unittest.TestCase):
    def test_vision_candidate_matching(self):
        people = [
            {"id": "S1", "embedding": [0.1]*128},
            {"id": "S2", "embedding": [0.8]*128},
        ]
        known = np.asarray([p["embedding"] for p in people], dtype=np.float64)
        test_vec = np.array([0.11]*128, dtype=np.float64)
        best = int(np.argmin(np.linalg.norm(known - test_vec, axis=1)))
        self.assertEqual(people[best]["id"], "S1")

    def test_offline_queue_read(self):
        tmp = tempfile.TemporaryDirectory()
        qf = Path(tmp.name) / "offline-events.jsonl"
        api.QUEUE_PATH = qf
        e1 = {"event_id": "off-1", "student_id": "S1"}
        e2 = {"event_id": "off-2", "student_id": "S2"}
        qf.write_text(f"{json.dumps(e1)}\n{json.dumps(e2)}\n", encoding="utf-8")
        pending = [json.loads(l) for l in qf.read_text(encoding="utf-8").splitlines() if l]
        self.assertEqual(len(pending), 2)
        self.assertEqual(pending[0]["event_id"], "off-1")
        tmp.cleanup()

    def test_ear_open_eye(self):
        p = np.array([[0,0],[1,1],[2,1],[3,0],[2,-1],[1,-1]], dtype=float)
        ear = (np.linalg.norm(p[1]-p[5]) + np.linalg.norm(p[2]-p[4])) / (2*np.linalg.norm(p[0]-p[3]))
        self.assertAlmostEqual(ear, 4/6, places=3)
        self.assertGreater(ear, 0.20)

    def test_ear_closed_eye(self):
        p = np.array([[0,0],[1,0.1],[2,0.1],[3,0],[2,-0.1],[1,-0.1]], dtype=float)
        ear = (np.linalg.norm(p[1]-p[5]) + np.linalg.norm(p[2]-p[4])) / (2*np.linalg.norm(p[0]-p[3]))
        self.assertLess(ear, 0.20)


# ══════════════════════════════════════════════════════════════════════════════
class TestAcademicAndScoping(unittest.TestCase):
    """Academic hierarchy, teacher scoping, class membership, and audit tracking."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        database.DATA_DIR = Path(self.tmp.name)
        database.DB_PATH = database.DATA_DIR / "presento.db"
        database.initialize()
        self.client = TestClient(app)

        from backend.database import connect
        with connect() as db:
            _seed_user(db, 1, "admin1", "adminpass", "admin")
            _seed_user(db, 2, "teacher1", "teachpass", "teacher")
            _seed_user(db, 3, "student1", "studpass", "student", "S1")
            _seed_user(db, 4, "student2", "studpass", "student", "S2")
            # Create students
            db.execute("INSERT INTO students(id, name, email, embedding) VALUES('S1', 'Student One', 's1@test.com', '[]')")
            db.execute("INSERT INTO students(id, name, email, embedding) VALUES('S2', 'Student Two', 's2@test.com', '[]')")
            # Create teacher profile for teacher1
            db.execute("INSERT INTO teacher_profiles(id, user_id, employee_id, designation) VALUES(1, 2, 'EMP-1', 'Professor')")

        self.admin_tok = self.client.post("/auth/login", data={"username": "admin1", "password": "adminpass"}).json()["access_token"]
        self.admin_h = {"Authorization": f"Bearer {self.admin_tok}"}
        self.teach_tok = self.client.post("/auth/login", data={"username": "teacher1", "password": "teachpass"}).json()["access_token"]
        self.teach_h = {"Authorization": f"Bearer {self.teach_tok}"}
        self.stud_tok = self.client.post("/auth/login", data={"username": "student1", "password": "studpass"}).json()["access_token"]
        self.stud_h = {"Authorization": f"Bearer {self.stud_tok}"}

    def tearDown(self):
        self.tmp.cleanup()

    def test_academic_hierarchy_and_scoping(self):
        # 1. Admin creates department, academic year, class, subject
        dept_res = self.client.post("/v1/academic/departments", headers=self.admin_h, json={"name": "Computer Science", "code": "CS"})
        self.assertEqual(dept_res.status_code, 201)
        dept_id = dept_res.json()["id"]

        ay_res = self.client.post("/v1/academic/academic-years", headers=self.admin_h, json={"label": "2026-2027", "is_current": True})
        self.assertEqual(ay_res.status_code, 201)
        ay_id = ay_res.json()["id"]

        cls_res = self.client.post("/v1/academic/classes", headers=self.admin_h, json={
            "department_id": dept_id, "academic_year_id": ay_id,
            "year": "TY", "semester": 5, "division": "A", "label": "TY CS-A"
        })
        self.assertEqual(cls_res.status_code, 201)
        cls_id = cls_res.json()["id"]

        subj_res = self.client.post("/v1/academic/subjects", headers=self.admin_h, json={
            "department_id": dept_id, "name": "Machine Learning", "code": "CS-501", "semester": 5
        })
        self.assertEqual(subj_res.status_code, 201)
        subj_id = subj_res.json()["id"]

        # 2. Assign teacher1 to subject & class
        assign_res = self.client.post("/v1/academic/teacher-assignments", headers=self.admin_h, json={
            "teacher_id": 1, "subject_id": subj_id, "class_id": cls_id
        })
        self.assertEqual(assign_res.status_code, 201)

        # 3. Assign student1 to class, student2 remains unassigned
        sp_res = self.client.post("/v1/academic/student-profiles", headers=self.admin_h, json={
            "student_id": "S1", "class_id": cls_id, "roll_number": "001"
        })
        self.assertEqual(sp_res.status_code, 201)

        # 4. Teacher checks my-assignments and my-classes
        my_assigns = self.client.get("/v1/teacher/my-assignments", headers=self.teach_h).json()
        self.assertEqual(len(my_assigns), 1)
        self.assertEqual(my_assigns[0]["class_label"], "TY CS-A")

        my_classes = self.client.get("/v1/teacher/my-classes", headers=self.teach_h).json()
        self.assertEqual(len(my_classes), 1)
        self.assertEqual(my_classes[0]["label"], "TY CS-A")

        # 5. Teacher scoping on session creation:
        # Cannot create session for unassigned class/subject (e.g. dummy class 999)
        now = datetime.now(timezone.utc).replace(microsecond=0)
        bad_sess = self.client.post("/v1/sessions", headers=self.teach_h, json={
            "room": "A101",
            "starts_at": (now - timedelta(minutes=5)).isoformat(),
            "ends_at": (now + timedelta(minutes=55)).isoformat(),
            "late_after_minutes": 10,
            "subject_id": subj_id,
            "class_id": 999
        })
        self.assertEqual(bad_sess.status_code, 403)

        # Can create session for assigned subject and class
        good_sess = self.client.post("/v1/sessions", headers=self.teach_h, json={
            "room": "A101",
            "starts_at": (now - timedelta(minutes=5)).isoformat(),
            "ends_at": (now + timedelta(minutes=55)).isoformat(),
            "late_after_minutes": 10,
            "subject_id": subj_id,
            "class_id": cls_id
        })
        self.assertEqual(good_sess.status_code, 201)
        sess_id = good_sess.json()["id"]

        # 6. Class membership validation on attendance:
        # student2 is not enrolled in TY CS-A -> 403 Forbidden
        bad_att = self.client.post("/v1/attendance/manual", headers=self.teach_h, json={
            "student_id": "S2",
            "session_id": sess_id,
            "status": "present",
            "observed_at": now.isoformat()
        })
        self.assertEqual(bad_att.status_code, 403)

        # student1 is enrolled in TY CS-A -> 201 Created
        good_att = self.client.post("/v1/attendance/manual", headers=self.teach_h, json={
            "student_id": "S1",
            "session_id": sess_id,
            "status": "present",
            "observed_at": now.isoformat()
        })
        self.assertEqual(good_att.status_code, 201)

        # 7. Edit attendance and verify audit tracking
        att_list = self.client.get(f"/v1/attendance?session_id={sess_id}", headers=self.teach_h).json()
        self.assertEqual(len(att_list), 1)
        att_id = att_list[0]["id"]

        patch_res = self.client.patch(f"/v1/attendance/{att_id}", headers=self.teach_h, json={
            "status": "late",
            "reason": "Traffic delay"
        })
        self.assertEqual(patch_res.status_code, 200)

        # Check updated record fields
        updated_att = self.client.get(f"/v1/attendance?session_id={sess_id}", headers=self.teach_h).json()[0]
        self.assertEqual(updated_att["status"], "late")
        self.assertEqual(updated_att["original_status"], "present")
        self.assertEqual(updated_att["modified_by"], "teacher1")

        # 8. Student view: subject-wise breakdown & enriched details
        stud_att = self.client.get("/v1/students/S1/attendance", headers=self.stud_h).json()
        self.assertEqual(stud_att["profile"]["class_label"], "TY CS-A")
        self.assertEqual(len(stud_att["subjects"]), 1)
        self.assertEqual(stud_att["subjects"][0]["subject_name"], "Machine Learning")
        self.assertEqual(stud_att["subjects"][0]["attended_sessions"], 1)

        # 9. Teacher closes session
        close_res = self.client.patch(f"/v1/sessions/{sess_id}/close", headers=self.teach_h)
        self.assertEqual(close_res.status_code, 200)

        # 10. Admin audit log inspection
        audit_res = self.client.get("/v1/admin/audit-log", headers=self.admin_h)
        self.assertEqual(audit_res.status_code, 200)
        logs = audit_res.json()
        self.assertGreater(len(logs), 0)



# ══════════════════════════════════════════════════════════════════════════════
class TestEnhancedAttendanceAndEdge(unittest.TestCase):
    """Tests for multi-student web check-in, duplicate handling, auto-absent, CSV exports, and edge vision matching."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        database.DATA_DIR = Path(self.tmp.name)
        database.DB_PATH = database.DATA_DIR / "presento.db"
        database.initialize()
        self.client = TestClient(app)

        self.s1_vec = np.zeros(128, dtype=np.float64)
        self.s2_vec = (np.zeros(128) + 5.0).astype(np.float64)

        from backend.database import connect
        with connect() as db:
            _seed_user(db, 1, "admin1", "adminpass", "admin")
            _seed_user(db, 2, "teacher1", "teachpass", "teacher")
            _seed_user(db, 3, "student1", "studpass", "student", "S1")
            _seed_user(db, 4, "student2", "studpass", "student", "S2")
            db.execute("INSERT INTO students(id, name, email, embedding) VALUES('S1', 'Student One', 's1@test.com', ?)", (json.dumps(self.s1_vec.tolist()),))
            db.execute("INSERT INTO students(id, name, email, embedding) VALUES('S2', 'Student Two', 's2@test.com', ?)", (json.dumps(self.s2_vec.tolist()),))
            db.execute("INSERT INTO teacher_profiles(id, user_id, employee_id, designation) VALUES(1, 2, 'EMP-1', 'Professor')")

        self.admin_tok = self.client.post("/auth/login", data={"username": "admin1", "password": "adminpass"}).json()["access_token"]
        self.admin_h = {"Authorization": f"Bearer {self.admin_tok}"}
        self.teach_tok = self.client.post("/auth/login", data={"username": "teacher1", "password": "teachpass"}).json()["access_token"]
        self.teach_h = {"Authorization": f"Bearer {self.teach_tok}"}
        self.stud_tok = self.client.post("/auth/login", data={"username": "student1", "password": "studpass"}).json()["access_token"]
        self.stud_h = {"Authorization": f"Bearer {self.stud_tok}"}

        # Setup basic academic hierarchy
        dept = self.client.post("/v1/academic/departments", headers=self.admin_h, json={"name": "Engineering", "code": "ENG"}).json()
        ay = self.client.post("/v1/academic/academic-years", headers=self.admin_h, json={"label": "2026-2027", "is_current": True}).json()
        cls = self.client.post("/v1/academic/classes", headers=self.admin_h, json={
            "department_id": dept["id"], "academic_year_id": ay["id"],
            "year": "TY", "division": "A", "label": "TY ENG-A"
        }).json()
        self.class_id = cls["id"]

        subj = self.client.post("/v1/academic/subjects", headers=self.admin_h, json={
            "department_id": dept["id"], "name": "Computer Vision", "code": "CV-101", "semester": 5
        }).json()
        self.subject_id = subj["id"]

        # Assign teacher to class and subject
        self.client.post("/v1/academic/teacher-assignments", headers=self.admin_h, json={
            "teacher_id": 1, "subject_id": self.subject_id, "class_id": self.class_id
        })

        # Enroll both S1 and S2 in the class
        self.client.post("/v1/academic/student-profiles", headers=self.admin_h, json={
            "student_id": "S1", "class_id": self.class_id, "roll_number": "101"
        })
        self.client.post("/v1/academic/student-profiles", headers=self.admin_h, json={
            "student_id": "S2", "class_id": self.class_id, "roll_number": "102"
        })

        # Create active session
        now = datetime.now(timezone.utc).replace(microsecond=0)
        sess = self.client.post("/v1/sessions", headers=self.teach_h, json={
            "room": "Lab 1",
            "starts_at": (now - timedelta(minutes=5)).isoformat(),
            "ends_at": (now + timedelta(minutes=55)).isoformat(),
            "late_after_minutes": 10,
            "subject_id": self.subject_id,
            "class_id": self.class_id
        }).json()
        self.session_id = sess["id"]

    def tearDown(self):
        self.tmp.cleanup()

    def test_role_boundaries_web_checkin(self):
        # 1. Student cannot perform web check-in (forbidden)
        res_stud = self.client.post("/v1/attendance/web-checkin", headers=self.stud_h, json={
            "images": ["dummy_base64"], "session_id": self.session_id
        })
        self.assertEqual(res_stud.status_code, 403)

        # 2. Teacher CAN perform web check-in
        from unittest.mock import patch
        with patch("backend.main.decode_image", return_value=np.zeros((10, 10, 3), dtype=np.uint8)), \
             patch("backend.main.embedding", return_value=np.zeros(128, dtype=np.float64)), \
             patch("backend.main.ear", return_value=0.25), \
             patch("backend.main.liveness_ok", return_value=True):
            res_teach = self.client.post("/v1/attendance/web-checkin", headers=self.teach_h, json={
                "images": ["dummy_base64"], "session_id": self.session_id
            })
            self.assertIn(res_teach.status_code, [200, 201])
            body = res_teach.json()
            self.assertEqual(body["status"], "present")
            self.assertFalse(body.get("already_present", False))

    def test_web_checkin_duplicate_and_auto_absent(self):
        from unittest.mock import patch
        with patch("backend.main.decode_image", return_value=np.zeros((10, 10, 3), dtype=np.uint8)), \
             patch("backend.main.embedding", return_value=np.zeros(128, dtype=np.float64)), \
             patch("backend.main.ear", return_value=0.25), \
             patch("backend.main.liveness_ok", return_value=True):
            # Check-in S1
            res1 = self.client.post("/v1/attendance/web-checkin", headers=self.teach_h, json={
                "images": ["dummy_base64"], "session_id": self.session_id
            })
            self.assertIn(res1.status_code, [200, 201])
            self.assertEqual(res1.json()["status"], "present")
            self.assertFalse(res1.json().get("already_present", False))

            # Check-in S1 AGAIN (duplicate) -> should return status: present and already_present: True without throwing error
            res2 = self.client.post("/v1/attendance/web-checkin", headers=self.teach_h, json={
                "images": ["dummy_base64"], "session_id": self.session_id
            })
            self.assertEqual(res2.status_code, 200)
            self.assertEqual(res2.json()["status"], "present")
            self.assertTrue(res2.json().get("already_present", False))

        # Finalize absent for unmarked students in session
        fin_res = self.client.post(f"/v1/sessions/{self.session_id}/finalize-absent", headers=self.teach_h)
        self.assertEqual(fin_res.status_code, 200)
        fin_body = fin_res.json()
        self.assertEqual(fin_body["marked_absent_count"], 1)

        # Verify attendance list
        records = self.client.get(f"/v1/attendance?session_id={self.session_id}", headers=self.teach_h).json()
        self.assertEqual(len(records), 2)
        rec_by_student = {r["student_id"]: r for r in records}
        self.assertEqual(rec_by_student["S1"]["status"], "present")
        self.assertEqual(rec_by_student["S2"]["status"], "absent")
        self.assertIn("absent on session", rec_by_student["S2"]["reason"].lower())

        # Repeated finalize-absent call is idempotent
        fin_res2 = self.client.post(f"/v1/sessions/{self.session_id}/finalize-absent", headers=self.teach_h)
        self.assertEqual(fin_res2.status_code, 200)
        self.assertEqual(fin_res2.json()["marked_absent_count"], 0)
        records_after = self.client.get(f"/v1/attendance?session_id={self.session_id}", headers=self.teach_h).json()
        self.assertEqual(len(records_after), 2)

    def test_csv_export_endpoints(self):
        from unittest.mock import patch
        with patch("backend.main.decode_image", return_value=np.zeros((10, 10, 3), dtype=np.uint8)), \
             patch("backend.main.embedding", return_value=np.zeros(128, dtype=np.float64)), \
             patch("backend.main.ear", return_value=0.25), \
             patch("backend.main.liveness_ok", return_value=True):
            # Create attendance record for S1
            self.client.post("/v1/attendance/web-checkin", headers=self.teach_h, json={
                "images": ["dummy_base64"], "session_id": self.session_id
            })

        # 1. Teacher/Admin CSV export
        exp_res = self.client.get(f"/v1/attendance/export?session_id={self.session_id}", headers=self.teach_h)
        self.assertEqual(exp_res.status_code, 200)
        self.assertIn("text/csv", exp_res.headers.get("content-type", ""))
        self.assertIn("Student ID", exp_res.text)
        self.assertIn("S1", exp_res.text)

        # 2. Student export own records
        s1_exp = self.client.get("/v1/students/S1/attendance/export", headers=self.stud_h)
        self.assertEqual(s1_exp.status_code, 200)
        self.assertIn("text/csv", s1_exp.headers.get("content-type", ""))
        self.assertIn("Date & Time", s1_exp.text)

        # 3. Student S1 cannot export student S2's records (forbidden)
        s2_exp = self.client.get("/v1/students/S2/attendance/export", headers=self.stud_h)
        self.assertEqual(s2_exp.status_code, 403)

    def test_edge_config_and_face_matching(self):
        from edge.config import settings
        from edge.vision import match_face_encoding

        # Test settings cooldown_seconds
        self.assertTrue(hasattr(settings, "cooldown_seconds"))
        self.assertIsInstance(settings.cooldown_seconds, float)
        self.assertGreater(settings.cooldown_seconds, 0.0)

        # Test match_face_encoding
        target_vec = np.random.RandomState(42).randn(128).astype(np.float64)
        people = [
            {"id": "S1", "name": "Student One", "embedding": target_vec.tolist()},
            {"id": "S2", "name": "Student Two", "embedding": json.dumps((target_vec + 5.0).tolist())},
        ]
        # Identical vector -> high confidence match
        match = match_face_encoding(target_vec, (10, 50, 50, 10), people, tolerance=0.6)
        self.assertIsNotNone(match)
        self.assertEqual(match.student_id, "S1")
        self.assertGreater(match.confidence, 0.9)

        # Far vector -> no match
        far_vec = (target_vec + 10.0).astype(np.float64)
        match_none = match_face_encoding(far_vec, (10, 50, 50, 10), people, tolerance=0.6)
        self.assertIsNone(match_none)


if __name__ == "__main__":
    unittest.main()


