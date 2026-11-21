"""
Database Seeding Script for SmartAttend AI.
Populates initial Admin, Faculty, Students, Academic entities,
and Sample Historic Attendance Records if empty.
"""

import time
import logging
from typing import Dict, Any

from backend.app.database.mongodb import db_manager
from backend.app.utils.security import hash_password
from backend.app.config import settings

logger = logging.getLogger("SmartAttend.Seed")


async def seed_initial_data():
    """Seeds baseline entities for instant out-of-the-box readiness."""
    user_col = db_manager.get_collection("users")
    user_count = await user_col.count_documents()
    if user_count > 0:
        logger.info("Database already seeded with %d users.", user_count)
        return

    logger.info("Seeding initial application data...")
    now = time.time()

    # 1. Users
    users = [
        {
            "id": "USR-ADMIN-01",
            "email": "admin@smartattend.ai",
            "password_hash": hash_password("admin123"),
            "full_name": "System Administrator",
            "role": "admin",
            "is_active": True,
            "created_at": now,
        },
        {
            "id": "USR-FAC-01",
            "email": "faculty@smartattend.ai",
            "password_hash": hash_password("faculty123"),
            "full_name": "Dr. Ramesh Kumar",
            "role": "faculty",
            "is_active": True,
            "created_at": now,
        },
        {
            "id": "USR-STU-01",
            "email": "student@smartattend.ai",
            "password_hash": hash_password("student123"),
            "full_name": "Aarav Sharma",
            "role": "student",
            "is_active": True,
            "created_at": now,
        },
    ]
    for u in users:
        await user_col.insert_one(u)

    # 2. Departments
    dept_col = db_manager.get_collection("departments")
    depts = [
        {"id": "DEP-CS", "name": "Computer Science & Engineering", "code": "CSE", "head": "Dr. A. Verma"},
        {"id": "DEP-EC", "name": "Electronics & Communication Engineering", "code": "ECE", "head": "Dr. S. Roy"},
        {"id": "DEP-ME", "name": "Mechanical Engineering", "code": "MECH", "head": "Dr. P. Patil"},
    ]
    for d in depts:
        await dept_col.insert_one(d)

    # 3. Classrooms
    cr_col = db_manager.get_collection("classrooms")
    classrooms = [
        {
            "id": "CR-101",
            "name": "Lecture Hall 101",
            "block": "Academic Block A",
            "capacity": 60,
            "geofence": {"latitude": 12.9716, "longitude": 77.5946, "radius_meters": 50},
        },
        {
            "id": "CR-102",
            "name": "Lecture Hall 102",
            "block": "Academic Block A",
            "capacity": 60,
            "geofence": {"latitude": 12.9718, "longitude": 77.5948, "radius_meters": 50},
        },
        {
            "id": "CR-LAB",
            "name": "AI & Computer Vision Lab 3",
            "block": "Technology Center",
            "capacity": 45,
            "geofence": {"latitude": 12.9720, "longitude": 77.5950, "radius_meters": 40},
        },
    ]
    for c in classrooms:
        await cr_col.insert_one(c)

    # 4. Subjects
    sub_col = db_manager.get_collection("subjects")
    subjects = [
        {
            "id": "SUB-CS301",
            "name": "Computer Vision & Biometrics",
            "code": "CS301",
            "department_id": "DEP-CS",
            "semester": 6,
            "credits": 4,
        },
        {
            "id": "SUB-CS302",
            "name": "Machine Learning & Deep Learning",
            "code": "CS302",
            "department_id": "DEP-CS",
            "semester": 6,
            "credits": 4,
        },
        {
            "id": "SUB-CS303",
            "name": "Distributed Database Systems",
            "code": "CS303",
            "department_id": "DEP-CS",
            "semester": 6,
            "credits": 3,
        },
    ]
    for s in subjects:
        await sub_col.insert_one(s)

    # 5. Faculty Profile
    fac_col = db_manager.get_collection("faculty")
    await fac_col.insert_one({
        "id": "FAC-101",
        "user_id": "USR-FAC-01",
        "employee_id": "EMP-2021-089",
        "full_name": "Dr. Ramesh Kumar",
        "email": "faculty@smartattend.ai",
        "department_id": "DEP-CS",
        "designation": "Associate Professor",
        "phone": "+91 98765 43210",
        "assigned_subjects": ["SUB-CS301", "SUB-CS302"],
        "assigned_classes": ["CSE-3A", "CSE-3B"],
    })

    # 6. Students
    stu_col = db_manager.get_collection("students")
    students = [
        {
            "id": "STU-2024-001",
            "user_id": "USR-STU-01",
            "student_id": "CS2401",
            "full_name": "Aarav Sharma",
            "email": "student@smartattend.ai",
            "department_id": "DEP-CS",
            "year": 3,
            "section": "A",
            "phone": "+91 91234 56789",
            "face_enrolled": True,
            "overall_attendance_pct": 88.5,
            "risk_status": "LOW_RISK",
        },
        {
            "id": "STU-2024-002",
            "user_id": None,
            "student_id": "CS2402",
            "full_name": "Diya Patel",
            "email": "diya.patel@student.smartattend.ai",
            "department_id": "DEP-CS",
            "year": 3,
            "section": "A",
            "phone": "+91 91234 56780",
            "face_enrolled": True,
            "overall_attendance_pct": 92.0,
            "risk_status": "LOW_RISK",
        },
        {
            "id": "STU-2024-003",
            "user_id": None,
            "student_id": "CS2403",
            "full_name": "Rohan Gupta",
            "email": "rohan.gupta@student.smartattend.ai",
            "department_id": "DEP-CS",
            "year": 3,
            "section": "A",
            "phone": "+91 91234 56781",
            "face_enrolled": True,
            "overall_attendance_pct": 68.0,
            "risk_status": "HIGH_RISK",
        },
        {
            "id": "STU-2024-004",
            "user_id": None,
            "student_id": "CS2404",
            "full_name": "Priya Nair",
            "email": "priya.nair@student.smartattend.ai",
            "department_id": "DEP-CS",
            "year": 3,
            "section": "A",
            "phone": "+91 91234 56782",
            "face_enrolled": True,
            "overall_attendance_pct": 74.0,
            "risk_status": "MODERATE_RISK",
        },
        {
            "id": "STU-2024-005",
            "user_id": None,
            "student_id": "CS2405",
            "full_name": "Vikram Singh",
            "email": "vikram.singh@student.smartattend.ai",
            "department_id": "DEP-CS",
            "year": 3,
            "section": "B",
            "phone": "+91 91234 56783",
            "face_enrolled": True,
            "overall_attendance_pct": 82.5,
            "risk_status": "LOW_RISK",
        },
    ]
    for s in students:
        await stu_col.insert_one(s)

    # 7. Timetables
    tt_col = db_manager.get_collection("timetables")
    timetables = [
        {
            "id": "TT-01",
            "subject_id": "SUB-CS301",
            "subject_name": "Computer Vision & Biometrics",
            "faculty_id": "FAC-101",
            "faculty_name": "Dr. Ramesh Kumar",
            "classroom_id": "CR-101",
            "classroom_name": "LH-101",
            "day_of_week": "Monday",
            "start_time": "09:00",
            "end_time": "10:00",
            "section": "A",
        },
        {
            "id": "TT-02",
            "subject_id": "SUB-CS302",
            "subject_name": "Machine Learning",
            "faculty_id": "FAC-101",
            "faculty_name": "Dr. Ramesh Kumar",
            "classroom_id": "CR-102",
            "classroom_name": "LH-102",
            "day_of_week": "Wednesday",
            "start_time": "11:00",
            "end_time": "12:00",
            "section": "A",
        },
        {
            "id": "TT-03",
            "subject_id": "SUB-CS301",
            "subject_name": "Computer Vision Lab",
            "faculty_id": "FAC-101",
            "faculty_name": "Dr. Ramesh Kumar",
            "classroom_id": "CR-LAB",
            "classroom_name": "AI Lab 3",
            "day_of_week": "Friday",
            "start_time": "14:00",
            "end_time": "16:00",
            "section": "A",
        },
    ]
    for t in timetables:
        await tt_col.insert_one(t)

    # 8. System Settings
    settings_col = db_manager.get_collection("system_settings")
    await settings_col.insert_one({
        "id": "SYS-SETTINGS-01",
        "recognition_threshold": settings.DEFAULT_RECOGNITION_THRESHOLD,
        "liveness_threshold": settings.DEFAULT_LIVENESS_THRESHOLD,
        "late_grace_minutes": settings.DEFAULT_LATE_GRACE_MINUTES,
        "email_alerts_enabled": settings.SMTP_ENABLED,
        "low_attendance_threshold_pct": 75.0,
        "require_liveness": True,
        "geofencing_enabled": True,
        "allow_manual_override": True,
    })

    # 9. Historic Attendance Sessions and Records for Analytics
    sess_col = db_manager.get_collection("attendance_sessions")
    rec_col = db_manager.get_collection("attendance_records")

    # Generate 5 recent sessions
    import datetime
    today = datetime.date.today()

    for day_offset in range(5, 0, -1):
        sess_date = (today - datetime.timedelta(days=day_offset)).strftime("%Y-%m-%d")
        sess_id = f"SESS-{sess_date}-CS301"
        sess_doc = {
            "id": sess_id,
            "subject_id": "SUB-CS301",
            "subject_name": "Computer Vision & Biometrics",
            "classroom_id": "CR-101",
            "faculty_id": "FAC-101",
            "faculty_name": "Dr. Ramesh Kumar",
            "date": sess_date,
            "start_time": "09:00:00",
            "end_time": "10:00:00",
            "status": "COMPLETED",
            "total_enrolled": 5,
            "present_count": 4 if day_offset != 3 else 3,
            "absent_count": 1 if day_offset != 3 else 2,
            "created_at": now - (day_offset * 86400),
        }
        await sess_col.insert_one(sess_doc)

        # Records
        for s in students:
            is_present = True
            is_late = False
            if s["id"] == "STU-2024-003" and day_offset in (1, 3, 4):
                is_present = False
            elif s["id"] == "STU-2024-004" and day_offset == 3:
                is_present = False
            elif s["id"] == "STU-2024-001" and day_offset == 2:
                is_late = True

            rec_doc = {
                "id": f"REC-{sess_id}-{s['id']}",
                "session_id": sess_id,
                "student_id": s["id"],
                "student_name": s["full_name"],
                "subject_id": "SUB-CS301",
                "date": sess_date,
                "timestamp": f"{sess_date} 09:04:12" if is_present else None,
                "status": "PRESENT" if is_present else "ABSENT",
                "is_late": is_late,
                "verification_method": "FACE_RECOGNITION" if is_present else "NONE",
                "confidence_score": 98.4 if is_present else 0.0,
                "distance_cm": 60 if is_present else 0,
            }
            await rec_col.insert_one(rec_doc)

    logger.info("Database successfully seeded with comprehensive initial demo data.")
