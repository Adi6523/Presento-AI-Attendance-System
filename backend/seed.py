"""
Synthetic demo database seeding utility.
Usage:
    python -m backend.seed
"""
from __future__ import annotations

import logging
from backend.database import SessionLocal
from backend.init_db import init_database
from backend.models import ClassModel, ClassStudent, Student, Teacher, User
from backend.services.auth_service import hash_password

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("presento.seed")


def seed_database() -> None:
    """Populates database with synthetic, privacy-safe demo records."""
    init_database()

    db = SessionLocal()
    try:
        logger.info("Starting synthetic demo seed...")

        # 1. Admin User
        admin_user = db.query(User).filter(User.username == "admin").first()
        if not admin_user:
            admin_user = User(
                username="admin",
                password_hash=hash_password("adminpassword123"),
                role="ADMIN",
                is_active=True,
            )
            db.add(admin_user)
            db.commit()
            db.refresh(admin_user)
            logger.info("Created Admin User: admin")
        else:
            logger.info("Admin user already exists.")

        # 2. Demo Teacher User & Profile
        teacher_user = db.query(User).filter(User.username == "demo_teacher").first()
        if not teacher_user:
            teacher_user = User(
                username="demo_teacher",
                password_hash=hash_password("teacherpassword123"),
                role="TEACHER",
                is_active=True,
            )
            db.add(teacher_user)
            db.commit()
            db.refresh(teacher_user)
            logger.info("Created Teacher User: demo_teacher")

        teacher = db.query(Teacher).filter(Teacher.user_id == teacher_user.id).first()
        if not teacher:
            teacher = Teacher(
                user_id=teacher_user.id,
                name="Prof. Demo Instructor",
                email="demo_teacher@example.edu",
            )
            db.add(teacher)
            db.commit()
            db.refresh(teacher)
            logger.info("Created Teacher Profile: %s", teacher.name)

        # 3. Synthetic Students (DEMO001, DEMO002)
        demo_students_data = [
            {
                "student_id": "DEMO001",
                "name": "Demo Student Alpha",
                "email": "demo001@example.edu",
                "department": "AIML",
                "year": "4th Year",
            },
            {
                "student_id": "DEMO002",
                "name": "Demo Student Beta",
                "email": "demo002@example.edu",
                "department": "AIML",
                "year": "4th Year",
            },
        ]

        students = []
        for s_data in demo_students_data:
            stu = db.query(Student).filter(Student.student_id == s_data["student_id"]).first()
            if not stu:
                stu = Student(
                    student_id=s_data["student_id"],
                    name=s_data["name"],
                    email=s_data["email"],
                    department=s_data["department"],
                    year=s_data["year"],
                    is_active=True,
                )
                db.add(stu)
                db.commit()
                db.refresh(stu)
                logger.info("Created Synthetic Student: %s (%s)", stu.name, stu.student_id)
            students.append(stu)

        # 4. Demo Class
        demo_class = db.query(ClassModel).filter(ClassModel.name == "Demo AIML").first()
        if not demo_class:
            demo_class = ClassModel(
                name="Demo AIML",
                subject="AIML401",
                teacher_id=teacher.id,
                academic_year="2026-2027",
                semester=7,
            )
            db.add(demo_class)
            db.commit()
            db.refresh(demo_class)
            logger.info("Created Demo Class: Demo AIML")

        # 5. Enrollments
        for stu in students:
            enrollment = (
                db.query(ClassStudent)
                .filter(ClassStudent.class_id == demo_class.id, ClassStudent.student_id == stu.id)
                .first()
            )
            if not enrollment:
                enrollment = ClassStudent(
                    class_id=demo_class.id,
                    student_id=stu.id,
                )
                db.add(enrollment)
                db.commit()
                logger.info("Enrolled %s into %s", stu.student_id, demo_class.name)

        logger.info("Synthetic demo seeding completed successfully.")
    finally:
        db.close()


if __name__ == "__main__":
    seed_database()
