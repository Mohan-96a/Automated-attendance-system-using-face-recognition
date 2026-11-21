"""
Student Management and Biometric Enrollment API Routes.
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File

from backend.app.schemas.student_schemas import (
    StudentCreateRequest,
    StudentUpdateRequest,
    StudentResponse,
    FaceEnrollmentApiRequest,
    FaceEnrollmentApiResponse,
)
from backend.app.services.student_service import StudentService
from backend.app.middleware.auth_middleware import get_current_user, RoleChecker
from backend.app.database.mongodb import db_manager

router = APIRouter(prefix="/students", tags=["Students"])


@router.get("", response_model=List[StudentResponse])
async def list_students(
    department_id: Optional[str] = None,
    year: Optional[int] = None,
    search: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(RoleChecker(["admin", "faculty"])),
):
    """Lists students with optional filters (Admin & Faculty only)."""
    return await StudentService.get_all_students(department_id, year, search)


@router.get("/{id}", response_model=StudentResponse)
async def get_student(
    id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Retrieves detailed profile for a specific student."""
    stu = await StudentService.get_student_by_id(id)
    if not stu:
        raise HTTPException(status_code=404, detail="Student not found")
    return stu


@router.post("", response_model=StudentResponse, status_code=status.HTTP_201_CREATED)
async def create_student(
    req: StudentCreateRequest,
    current_user: Dict[str, Any] = Depends(RoleChecker(["admin"])),
):
    """Creates a new student record (Admin only)."""
    return await StudentService.create_student(req.model_dump(), current_user)


@router.post("/{id}/enroll-face", response_model=FaceEnrollmentApiResponse)
async def enroll_student_face(
    id: str,
    req: FaceEnrollmentApiRequest,
    current_user: Dict[str, Any] = Depends(RoleChecker(["admin", "faculty", "student"])),
):
    """
    Biometric enrollment endpoint:
    Processes facial image, checks quality and liveness, extracts 512-d ArcFace embedding,
    encrypts it with AES/Fernet, and activates face recognition for attendance.
    """
    # If student role, they can only enroll their own face
    if current_user.get("role") == "student":
        stu = await StudentService.get_student_by_id(id)
        if not stu or stu.get("user_id") != current_user.get("id"):
            raise HTTPException(status_code=403, detail="Students can only enroll their own face biometrics.")

    return await StudentService.enroll_face(
        student_id=id,
        image_base64=req.image_base64,
        actor_user=current_user,
        enforce_liveness=True,
    )


@router.delete("/{id}/biometrics")
async def delete_student_biometrics(
    id: str,
    current_user: Dict[str, Any] = Depends(RoleChecker(["admin"])),
):
    """GDPR-compliant biometric data erasure endpoint."""
    return await StudentService.delete_biometrics(id, current_user)


@router.post("/bulk-import")
async def bulk_import(
    file: UploadFile = File(...),
    current_user: Dict[str, Any] = Depends(RoleChecker(["admin"])),
):
    """Bulk imports students from CSV or Excel file."""
    content = await file.read()
    return await StudentService.bulk_import_students(content, file.filename, current_user)
