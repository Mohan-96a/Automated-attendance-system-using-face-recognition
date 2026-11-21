"""
Pydantic Schemas for Student and Faculty Management.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


# --- Student Schemas ---
class StudentCreateRequest(BaseModel):
    student_id: str = Field(description="Unique Roll Number or Registration ID")
    full_name: str
    email: str
    department_id: str
    year: int = 1
    section: str = "A"
    phone: Optional[str] = None


class StudentUpdateRequest(BaseModel):
    full_name: Optional[str] = None
    email: Optional[str] = None
    department_id: Optional[str] = None
    year: Optional[int] = None
    section: Optional[str] = None
    phone: Optional[str] = None
    is_active: Optional[bool] = None


class StudentResponse(BaseModel):
    id: str
    student_id: str
    full_name: str
    email: str
    department_id: str
    year: int
    section: str
    phone: Optional[str] = None
    face_enrolled: bool = False
    overall_attendance_pct: float = 0.0
    risk_status: str = "LOW_RISK"
    created_at: Optional[float] = None


class FaceEnrollmentApiRequest(BaseModel):
    image_base64: str = Field(description="Base64 encoded JPEG/PNG image")


class FaceEnrollmentApiResponse(BaseModel):
    success: bool
    message: str
    student_id: str
    face_enrolled: bool
    quality_score: float
    liveness_score: float
    thumbnail_base64: Optional[str] = None


# --- Faculty Schemas ---
class FacultyCreateRequest(BaseModel):
    employee_id: str
    full_name: str
    email: str
    department_id: str
    designation: str = "Assistant Professor"
    phone: Optional[str] = None
    assigned_subjects: List[str] = Field(default_factory=list)
    assigned_classes: List[str] = Field(default_factory=list)


class FacultyUpdateRequest(BaseModel):
    full_name: Optional[str] = None
    email: Optional[str] = None
    department_id: Optional[str] = None
    designation: Optional[str] = None
    phone: Optional[str] = None
    assigned_subjects: Optional[List[str]] = None
    assigned_classes: Optional[List[str]] = None


class FacultyResponse(BaseModel):
    id: str
    employee_id: str
    full_name: str
    email: str
    department_id: str
    designation: str
    phone: Optional[str] = None
    assigned_subjects: List[str] = Field(default_factory=list)
    assigned_classes: List[str] = Field(default_factory=list)
    created_at: Optional[float] = None
