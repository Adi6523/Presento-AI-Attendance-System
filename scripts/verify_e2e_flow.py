"""
Presento Phase 3 — End-to-End Frontend Contract & Backend Flow Verification
Validates the entire lifecycle required by Section 33 of User Request:
1. Teacher authentication (/api/auth/login)
2. Identity verification (/api/auth/me)
3. Dashboard loading (/api/dashboard/summary, recent-attendance)
4. Course loading (/api/classes)
5. Course roster & enrollment (/api/classes/{id}/students/{student_id})
6. Session creation & start (/api/sessions, /api/sessions/{id}/start)
7. AI Attendance event emission (/api/attendance/events)
8. Frontend polling verification (/api/sessions/{id}/attendance)
9. Duplicate idempotency check
10. Session closure (/api/sessions/{id}/close)
11. Automatic ABSENT record generation verification
12. Attendance history verification (/api/attendance/history)
13. Logout (/api/auth/logout)
"""

import os
import sys
from datetime import datetime, timezone
from fastapi.testclient import TestClient

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.main import app
from backend.database import SessionLocal
from backend.init_db import init_database
from backend.models import User, Student, ClassModel, ClassStudent, Teacher
from backend.services.auth_service import hash_password

def run_e2e():
    print("=== STARTING PRESENTO PHASE 3 E2E INTEGRATION SUITE ===")
    init_database()
    client = TestClient(app)

    db = SessionLocal()
    try:
        # Seed teacher prof_patil if not exists
        teacher_user = db.query(User).filter(User.username == "prof_patil").first()
        if not teacher_user:
            teacher_user = User(
                username="prof_patil",
                password_hash=hash_password("ProfSecure123!"),
                role="TEACHER",
                is_active=True
            )
            db.add(teacher_user)
            db.commit()
            db.refresh(teacher_user)
            teacher = Teacher(user_id=teacher_user.id, name="Prof. Patil")
            db.add(teacher)
            db.commit()

        # Seed students
        s1 = db.query(Student).filter(Student.student_id == "23UAM090").first()
        if not s1:
            s1 = Student(student_id="23UAM090", name="Aditya Patil", department="AI & DS", year="Year 4", is_active=True)
            db.add(s1)
        s2 = db.query(Student).filter(Student.student_id == "23UAM092").first()
        if not s2:
            s2 = Student(student_id="23UAM092", name="Piyush Sharma", department="AI & DS", year="Year 4", is_active=True)
            db.add(s2)
        db.commit()
    finally:
        db.close()

    # Step 1: Login with seeded teacher
    print("\n[Step 1] Teacher Login (/api/auth/login)")
    login_resp = client.post("/api/auth/login", json={"username": "prof_patil", "password": "ProfSecure123!"})
    assert login_resp.status_code == 200, f"Login failed: {login_resp.text}"
    token_data = login_resp.json()
    token = token_data["access_token"]
    assert token, "Token missing in response"
    auth_headers = {"Authorization": f"Bearer {token}"}
    print(f"  [OK] Login successful. Token received: {token[:16]}...")

    # Step 2: Get Auth Profile
    print("\n[Step 2] Verify Current User (/api/auth/me)")
    me_resp = client.get("/api/auth/me", headers=auth_headers)
    assert me_resp.status_code == 200, f"Get me failed: {me_resp.text}"
    user_info = me_resp.json()
    assert user_info["username"] == "prof_patil"
    print(f"  [OK] User verified: {user_info['username']} (Role: {user_info['role']})")

    # Step 3: Dashboard Summary & Activity
    print("\n[Step 3] Dashboard APIs (/api/dashboard/summary, recent-attendance)")
    summary_resp = client.get("/api/dashboard/summary", headers=auth_headers)
    assert summary_resp.status_code == 200
    summary = summary_resp.json()
    print(f"  [OK] Summary: total_students={summary['total_students']}, total_classes={summary['total_classes']}")

    recent_resp = client.get("/api/dashboard/recent-attendance", headers=auth_headers)
    assert recent_resp.status_code == 200
    print(f"  [OK] Recent attendance fetched successfully.")

    # Step 4: Course Management
    print("\n[Step 4] Course Management (/api/classes)")
    cls_payload = {
        "name": f"Computer Vision {datetime.now().strftime('%M%S')}",
        "subject": "CS401",
        "academic_year": "2026-2027",
        "semester": 7
    }
    create_cls_resp = client.post("/api/classes", json=cls_payload, headers=auth_headers)
    assert create_cls_resp.status_code == 201, f"Create class failed: {create_cls_resp.text}"
    cls_data = create_cls_resp.json()
    class_id = cls_data["id"]
    print(f"  [OK] Created class: '{cls_data['name']}' (ID: {class_id})")

    # Step 5: Enroll Students into Course
    print("\n[Step 5] Enroll Students (/api/classes/{id}/students/{student_id})")
    e1 = client.post(f"/api/classes/{class_id}/students/23UAM090", headers=auth_headers)
    assert e1.status_code in (200, 201), f"Enrollment 1 failed: {e1.text}"
    e2 = client.post(f"/api/classes/{class_id}/students/23UAM092", headers=auth_headers)
    assert e2.status_code in (200, 201), f"Enrollment 2 failed: {e2.text}"

    class_view = client.get(f"/api/classes/{class_id}", headers=auth_headers)
    assert class_view.status_code == 200
    students_in_cls = class_view.json()["students"]
    assert len(students_in_cls) == 2
    print(f"  [OK] Enrolled 2 students in class: {[s['student_id'] for s in students_in_cls]}")

    # Step 6: Create & Start Attendance Session
    print("\n[Step 6] Create & Start Attendance Session")
    sess_payload = {
        "class_id": class_id,
        "date": datetime.now().strftime("%Y-%m-%d")
    }
    sess_create = client.post("/api/sessions", json=sess_payload, headers=auth_headers)
    assert sess_create.status_code == 201, f"Session create failed: {sess_create.text}"
    session_id = sess_create.json()["id"]
    print(f"  [OK] Session created: ID={session_id}, status={sess_create.json()['status']}")

    sess_start = client.post(f"/api/sessions/{session_id}/start", headers=auth_headers)
    assert sess_start.status_code == 200, f"Session start failed: {sess_start.text}"
    assert sess_start.json()["status"] == "ACTIVE"
    print(f"  [OK] Session started: status={sess_start.json()['status']}")

    # Step 7: Initial Frontend Poll (Empty Attendance)
    print("\n[Step 7] Frontend Poll Initial Session Attendance (/api/sessions/{id}/attendance)")
    poll_1 = client.get(f"/api/sessions/{session_id}/attendance", headers=auth_headers)
    assert poll_1.status_code == 200
    assert len(poll_1.json()) == 0
    print(f"  [OK] Initial attendance count: {len(poll_1.json())}")

    # Step 8: AI System emits verified attendance event
    print("\n[Step 8] AI Camera sends attendance event (/api/attendance/events)")
    from backend.config import settings
    ai_event_payload = {
        "session_id": session_id,
        "student_id": "23UAM090",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "confidence": 0.94,
        "liveness_state": "REAL_LOCKED",
        "liveness_score": 0.98,
        "device_id": "raspi_cam_01",
        "source": "AI_CAMERA"
    }
    ai_headers = {"Authorization": f"Bearer {settings.presento_ai_api_key}"}
    ai_resp = client.post("/api/attendance/events", json=ai_event_payload, headers=ai_headers)
    assert ai_resp.status_code == 200, f"AI event failed: {ai_resp.text}"
    ai_res_data = ai_resp.json()
    assert ai_res_data["status"] == "PRESENT"
    print(f"  [OK] AI Event admitted: status={ai_res_data['status']}, event_id={ai_res_data.get('event_id')}")

    # Step 9: Frontend 2-3s Poll verifies PRESENT record
    print("\n[Step 9] Frontend Polls Live Attendance")
    poll_2 = client.get(f"/api/sessions/{session_id}/attendance", headers=auth_headers)
    assert poll_2.status_code == 200
    att_records = poll_2.json()
    assert len(att_records) == 1
    rec1 = att_records[0]
    assert rec1["student_id"] == "23UAM090"
    assert rec1["status"] == "PRESENT"
    assert rec1["confidence"] == 0.94
    assert rec1["liveness_score"] == 0.98
    print(f"  [OK] Record received: Student {rec1['student_id']} PRESENT, Liveness score={rec1['liveness_score']}")

    # Step 10: Deduplication check
    print("\n[Step 10] AI event deduplication check")
    ai_resp_dup = client.post("/api/attendance/events", json=ai_event_payload, headers=ai_headers)
    assert ai_resp_dup.status_code == 200
    assert ai_resp_dup.json()["status"] == "DUPLICATE"
    poll_dup = client.get(f"/api/sessions/{session_id}/attendance", headers=auth_headers)
    assert len(poll_dup.json()) == 1, "Duplicate record erroneously created in attendance!"
    print(f"  [OK] Deduplication verified. Event: DUPLICATE, Total records: {len(poll_dup.json())}")

    # Step 11: Close Session
    print("\n[Step 11] Teacher closes session (/api/sessions/{id}/close)")
    close_resp = client.post(f"/api/sessions/{session_id}/close", headers=auth_headers)
    assert close_resp.status_code == 200, f"Session close failed: {close_resp.text}"
    assert close_resp.json()["status"] == "COMPLETED"
    print(f"  [OK] Session successfully closed: status={close_resp.json()['status']}")

    # Step 12: Verify Absent Records Generation
    print("\n[Step 12] Verify Backend Generated ABSENT records on close")
    poll_closed = client.get(f"/api/sessions/{session_id}/attendance", headers=auth_headers)
    assert poll_closed.status_code == 200
    final_records = poll_closed.json()
    assert len(final_records) == 2, f"Expected 2 records (1 PRESENT, 1 ABSENT), got {len(final_records)}"

    rec_by_id = {r["student_id"]: r for r in final_records}
    assert rec_by_id["23UAM090"]["status"] == "PRESENT"
    assert rec_by_id["23UAM092"]["status"] == "ABSENT"
    print(f"  [OK] 23UAM090: {rec_by_id['23UAM090']['status']}")
    print(f"  [OK] 23UAM092: {rec_by_id['23UAM092']['status']} (Backend generated)")

    # Step 13: Attendance History
    print("\n[Step 13] Verify Attendance History (/api/attendance/history)")
    hist_resp = client.get(f"/api/attendance/history?class_id={class_id}", headers=auth_headers)
    assert hist_resp.status_code == 200
    hist_data = hist_resp.json()
    assert isinstance(hist_data, list)
    assert len(hist_data) >= 2
    print(f"  [OK] History contains {len(hist_data)} entries for this class.")

    # Step 14: Teacher Logout
    print("\n[Step 14] Teacher Logout (/api/auth/logout)")
    logout_resp = client.post("/api/auth/logout", headers=auth_headers)
    assert logout_resp.status_code == 200
    print(f"  [OK] Logged out successfully: {logout_resp.json()}")

    print("\n=======================================================")
    print("[PASS] ALL 14 END-TO-END CRITICAL VERIFICATIONS PASSED!")
    print("=======================================================")

if __name__ == "__main__":
    run_e2e()
