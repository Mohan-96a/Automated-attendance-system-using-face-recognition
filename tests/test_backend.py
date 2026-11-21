"""
Comprehensive Test Suite for SmartAttend AI Backend, Database, and Authentication (Phase 2).
"""

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.utils.security import hash_password, verify_password, create_access_token, decode_access_token
from ai_engine.evaluation.evaluate_models import generate_realistic_test_face
import cv2
import base64


@pytest.fixture(scope="module")
def client():
    """Contextual TestClient that automatically invokes FastAPI lifespan and seeds data."""
    with TestClient(app) as c:
        yield c


def test_password_security():
    """Tests password hashing and salted PBKDF2 verification."""
    password = "MySecurePassword123!"
    hashed = hash_password(password)
    assert verify_password(password, hashed) is True
    assert verify_password("WrongPassword", hashed) is False


def test_jwt_token_generation():
    """Tests JWT encoding and decoding."""
    payload = {"sub": "USR-TEST-01", "role": "admin", "email": "test@smartattend.ai"}
    token = create_access_token(payload)
    decoded = decode_access_token(token)
    assert decoded is not None
    assert decoded["sub"] == "USR-TEST-01"
    assert decoded["role"] == "admin"


def test_root_and_health_endpoints(client):
    """Tests public health endpoints."""
    res = client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert data["product"] == "SmartAttend AI"

    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "healthy"


def test_admin_login(client):
    """Tests admin authentication with seeded credentials."""
    res = client.post(
        "/api/v1/auth/login",
        json={"email": "admin@smartattend.ai", "password": "admin123"},
    )
    assert res.status_code == 200
    data = res.json()
    assert "access_token" in data
    assert data["role"] == "admin"
    assert data["email"] == "admin@smartattend.ai"


def test_faculty_login(client):
    """Tests faculty authentication with seeded credentials."""
    res = client.post(
        "/api/v1/auth/login",
        json={"email": "faculty@smartattend.ai", "password": "faculty123"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["role"] == "faculty"


def test_student_login(client):
    """Tests student authentication with seeded credentials."""
    res = client.post(
        "/api/v1/auth/login",
        json={"email": "student@smartattend.ai", "password": "student123"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["role"] == "student"


def test_unauthorized_access_fails(client):
    """Tests that protected routes reject unauthorized requests."""
    res = client.get("/api/v1/students")
    assert res.status_code == 401


def test_student_management_apis(client):
    """Tests student listing and details with admin token."""
    login_res = client.post(
        "/api/v1/auth/login",
        json={"email": "admin@smartattend.ai", "password": "admin123"},
    )
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # List students
    res = client.get("/api/v1/students", headers=headers)
    assert res.status_code == 200
    students = res.json()
    assert len(students) >= 5

    # Get student by ID
    s_id = students[0]["id"]
    res_single = client.get(f"/api/v1/students/{s_id}", headers=headers)
    assert res_single.status_code == 200
    assert res_single.json()["id"] == s_id


def test_academic_apis(client):
    """Tests academic departments, classrooms, subjects, timetables."""
    login_res = client.post(
        "/api/v1/auth/login",
        json={"email": "admin@smartattend.ai", "password": "admin123"},
    )
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Departments
    res_dept = client.get("/api/v1/academics/departments", headers=headers)
    assert res_dept.status_code == 200
    assert len(res_dept.json()) >= 3

    # Classrooms
    res_cr = client.get("/api/v1/academics/classrooms", headers=headers)
    assert res_cr.status_code == 200
    assert len(res_cr.json()) >= 3

    # Subjects
    res_sub = client.get("/api/v1/academics/subjects", headers=headers)
    assert res_sub.status_code == 200
    assert len(res_sub.json()) >= 3

    # Timetables
    res_tt = client.get("/api/v1/academics/timetables", headers=headers)
    assert res_tt.status_code == 200
    assert len(res_tt.json()) >= 3


def test_attendance_session_flow(client):
    """Tests creating, querying, and updating an attendance session."""
    login_res = client.post(
        "/api/v1/auth/login",
        json={"email": "faculty@smartattend.ai", "password": "faculty123"},
    )
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 1. Create Session
    create_payload = {
        "subject_id": "SUB-CS301",
        "classroom_id": "CR-101",
        "faculty_id": "FAC-101",
        "start_time": "09:00:00",
        "end_time": "10:00:00",
        "late_grace_minutes": 15,
    }
    res_create = client.post("/api/v1/attendance/sessions", json=create_payload, headers=headers)
    assert res_create.status_code == 201
    session = res_create.json()
    sess_id = session["id"]
    assert session["status"] == "ACTIVE"

    # 2. Get active sessions
    res_active = client.get("/api/v1/attendance/sessions/active", headers=headers)
    assert res_active.status_code == 200
    assert any(s["id"] == sess_id for s in res_active.json())

    # 3. Manual mark attendance override
    manual_payload = {
        "student_id": "STU-2024-001",
        "status": "PRESENT",
        "is_late": False,
        "notes": "Verified present by instructor",
    }
    res_manual = client.post(f"/api/v1/attendance/sessions/{sess_id}/manual", json=manual_payload, headers=headers)
    assert res_manual.status_code == 200

    # 4. Fetch session records
    res_recs = client.get(f"/api/v1/attendance/sessions/{sess_id}/records", headers=headers)
    assert res_recs.status_code == 200
    records = res_recs.json()
    marked_student = next(r for r in records if r["student_id"] == "STU-2024-001")
    assert marked_student["status"] == "PRESENT"

    # 5. Stop session
    res_stop = client.put(f"/api/v1/attendance/sessions/{sess_id}/status?status=COMPLETED", headers=headers)
    assert res_stop.status_code == 200
    assert res_stop.json()["status"] == "COMPLETED"


def test_ai_frame_processing_endpoint(client):
    """Tests the real-time AI live frame processing endpoint."""
    login_res = client.post(
        "/api/v1/auth/login",
        json={"email": "faculty@smartattend.ai", "password": "faculty123"},
    )
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Create active session
    create_payload = {
        "subject_id": "SUB-CS302",
        "classroom_id": "CR-102",
        "faculty_id": "FAC-101",
        "start_time": "11:00:00",
        "end_time": "12:00:00",
        "late_grace_minutes": 10,
    }
    sess_res = client.post("/api/v1/attendance/sessions", json=create_payload, headers=headers)
    sess_id = sess_res.json()["id"]

    # Generate synthetic frame with realistic test face
    test_img = generate_realistic_test_face(seed=42)
    _, buf = cv2.imencode(".jpg", test_img)
    b64_frame = base64.b64encode(buf).decode("utf-8")

    proc_payload = {
        "session_id": sess_id,
        "frame_base64": b64_frame,
        "user_latitude": 12.9718,
        "user_longitude": 77.5948,
    }

    res_proc = client.post(f"/api/v1/attendance/sessions/{sess_id}/process-frame", json=proc_payload, headers=headers)
    assert res_proc.status_code == 200
    proc_data = res_proc.json()
    assert proc_data["success"] is True
    assert proc_data["faces_detected"] >= 1
    assert "annotated_frame_base64" in proc_data
    assert proc_data["processing_time_ms"] > 0
