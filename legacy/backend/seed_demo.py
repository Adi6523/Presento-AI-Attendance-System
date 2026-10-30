"""
Seeds the database with demo accounts, academic structure, and sample sessions.
Run once: python -m backend.seed_demo
Safe to re-run — uses INSERT OR IGNORE throughout.
"""
from datetime import datetime, timedelta, timezone
from .database import connect, initialize
from .auth import hash_password

initialize()

now = datetime.now(timezone.utc).replace(second=0, microsecond=0, tzinfo=None)

with connect() as db:

    # ══════════════════════════════════════════════════════════════════════════
    # 1. ACADEMIC STRUCTURE
    # ══════════════════════════════════════════════════════════════════════════

    # ── Department ─────────────────────────────────────────────────────────
    db.execute(
        "INSERT OR IGNORE INTO departments(id, name, code) VALUES(?, ?, ?)",
        (1, "Computer Science & Engineering (AI & ML)", "CSE-AIML")
    )

    # ── Academic year ──────────────────────────────────────────────────────
    db.execute(
        "INSERT OR IGNORE INTO academic_years(id, label, is_current) VALUES(?, ?, ?)",
        (1, "2026-2027", 1)
    )

    # ── Class / Division ───────────────────────────────────────────────────
    db.execute(
        "INSERT OR IGNORE INTO classes(id, department_id, academic_year_id, year, semester, division, label) "
        "VALUES(?, ?, ?, ?, ?, ?, ?)",
        (1, 1, 1, "TY", 5, "A", "TY CSE-AIML A")
    )

    # ── Subjects (match existing course_code values) ──────────────────────
    demo_subjects = [
        (1, 1, "Machine Learning",               "CSE-AIML-401", 5),
        (2, 1, "Deep Learning Lab",               "CSE-AIML-402", 5),
        (3, 1, "Natural Language Processing",     "CSE-AIML-403", 5),
        (4, 1, "Computer Vision",                 "CSE-AIML-404", 5),
    ]
    for sid, dept, name, code, sem in demo_subjects:
        db.execute(
            "INSERT OR IGNORE INTO subjects(id, department_id, name, code, semester) VALUES(?,?,?,?,?)",
            (sid, dept, name, code, sem)
        )

    # ══════════════════════════════════════════════════════════════════════════
    # 2. USER ACCOUNTS (Faculty & Administration)
    # ══════════════════════════════════════════════════════════════════════════

    # (username, password, email, role, student_id)
    accounts = [
        ("admin",    "admin123",    "admin@presento.local",   "admin",   None),
        ("teacher1", "teacher123",  "teacher@presento.local", "teacher", None),
    ]
    for username, password, email, role, sid in accounts:
        db.execute(
            "INSERT OR IGNORE INTO users(username, email, password_hash, role, student_id) "
            "VALUES(?,?,?,?,?)",
            (username, email, hash_password(password), role, sid)
        )

    # ══════════════════════════════════════════════════════════════════════════
    # 3. TEACHER PROFILE + ASSIGNMENTS
    # ══════════════════════════════════════════════════════════════════════════

    # Look up teacher1's user ID
    teacher_user = db.execute("SELECT id FROM users WHERE username='teacher1'").fetchone()
    if teacher_user:
        teacher_uid = teacher_user[0]

        # Create teacher profile
        db.execute(
            "INSERT OR IGNORE INTO teacher_profiles(id, user_id, department_id, employee_id, designation) "
            "VALUES(?,?,?,?,?)",
            (1, teacher_uid, 1, "EMP-001", "Assistant Professor")
        )

        # Assign teacher1 to all 4 subjects for TY CSE-AIML A
        for subj_id in [1, 2, 3, 4]:
            db.execute(
                "INSERT OR IGNORE INTO teacher_assignments(teacher_id, subject_id, class_id) "
                "VALUES(?,?,?)",
                (1, subj_id, 1)  # teacher_profile.id=1, class.id=1
            )

    # ══════════════════════════════════════════════════════════════════════════
    # 4. DEMO SESSIONS (with academic context)
    # ══════════════════════════════════════════════════════════════════════════

    # Look up teacher profile ID
    tp = db.execute("SELECT id FROM teacher_profiles WHERE user_id=?",
                    (teacher_uid,)).fetchone() if teacher_user else None
    tp_id = tp[0] if tp else None

    demo_sessions = [
        ("CSE-AIML-401", "A101",
         (now - timedelta(hours=1)).isoformat(),
         (now + timedelta(hours=1)).isoformat(),
         10, tp_id, 1, 1, "open"),   # Machine Learning, class 1, subject 1
        ("CSE-AIML-402", "Lab-2",
         (now + timedelta(hours=2)).isoformat(),
         (now + timedelta(hours=4)).isoformat(),
         15, tp_id, 2, 1, "open"),   # Deep Learning Lab
        ("CSE-AIML-403", "Seminar Hall",
         (now + timedelta(hours=4)).isoformat(),
         (now + timedelta(hours=6)).isoformat(),
         10, tp_id, 3, 1, "open"),   # NLP
        ("CSE-AIML-404", "B204",
         (now + timedelta(days=1)).replace(hour=10, minute=0).isoformat(),
         (now + timedelta(days=1)).replace(hour=12, minute=0).isoformat(),
         10, tp_id, 4, 1, "open"),   # Computer Vision
    ]
    for code, room, start, end, late, tid, subj, cls, status in demo_sessions:
        db.execute(
            "INSERT OR IGNORE INTO sessions"
            "(course_code, room, starts_at, ends_at, late_after_minutes, teacher_id, subject_id, class_id, status) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (code, room, start, end, late, tid, subj, cls, status)
        )


print()
print("  Presento demo data seeded!")
print()
print("""  +------------------------------------------------------------+
  |  DEMO ACCOUNTS                                            |
  |------------+--------------+----------+---------------------|
  | Username   | Password     | Role     | Notes               |
  |------------+--------------+----------+---------------------|
  | admin      | admin123     | Admin    | Full access         |
  | teacher1   | teacher123   | Teacher  | TY CSE-AIML A       |
  |------------+--------------+----------+---------------------|

  ACADEMIC STRUCTURE:
    Department : Computer Science & Engineering (AI & ML)
    Class      : TY CSE-AIML A  (2026-2027, Semester 5)
    Subjects   : CSE-AIML-401 Machine Learning
    	       CSE-AIML-402 Deep Learning Lab
    	       CSE-AIML-403 Natural Language Processing
    	       CSE-AIML-404 Computer Vision
    Teacher    : teacher1 -> assigned to all 4 subjects
    Students   : (Ready for admin to add real students)

  [!] Change passwords before using in production!""")
print()
