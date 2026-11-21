"""
Pydantic Schemas for Attendance Sessions, Records, and Corrections.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


# --- Session Schemas ---
class CreateSessionRequest(BaseModel):
    subject_id: str
    classroom_id: str
    faculty_id: str
    date: Optional[str] = None  # YYYY-MM-DD (defaults to today)
    start_time: str = "09:00:00"
    end_time: str = "10:00:00"
    late_grace_minutes: int = 10


class AttendanceSessionResponse(BaseModel):
    id: str
    subject_id: str
    subject_name: Optional[str] = None
    classroom_id: str
    classroom_name: Optional[str] = None
    faculty_id: str
    faculty_name: Optional[str] = None
    date: str
    start_time: str
    end_time: str
    status: str  # "ACTIVE", "PAUSED", "COMPLETED"
    total_enrolled: int = 0
    present_count: int = 0
    absent_count: int = 0
    created_at: float


# --- Attendance Record Schemas ---
class AttendanceRecordResponse(BaseModel):
    id: str
    session_id: str
    student_id: str
    student_name: str
    subject_id: str
    date: str
    timestamp: Optional[str] = None
    status: str  # "PRESENT", "ABSENT"
    is_late: bool = False
    verification_method: str = "FACE_RECOGNITION"
    confidence_score: float = 0.0
    distance_cm: int = 60


class ManualAttendanceRequest(BaseModel):
    student_id: str
    status: str = "PRESENT"  # "PRESENT" or "ABSENT"
    is_late: bool = False
    notes: Optional[str] = None


class ProcessFrameRequest(BaseModel):
    frame_base64: str = Field(description="Base64 encoded JPEG/PNG frame from webcam")
    session_id: str
    user_latitude: Optional[float] = None
    user_longitude: Optional[float] = None


class ProcessFrameResponse(BaseModel):
    success: bool
    faces_detected: int
    multiple_faces_alert: bool
    annotated_frame_base64: Optional[str] = None
    recognized_students: List[Dict[str, Any]]
    processing_time_ms: float
    fps: float


# --- Correction Request Schemas ---
class CreateCorrectionRequest(BaseModel):
    session_id: str
    student_id: str
    reason: str
    requested_status: str = "PRESENT"


class CorrectionRequestResponse(BaseModel):
    id: str
    session_id: str
    student_id: str
    student_name: str
    date: str
    reason: str
    requested_status: str
    status: str  # "PENDING", "APPROVED", "REJECTED"
    faculty_notes: Optional[str] = None
    created_at: float


class ReviewCorrectionRequest(BaseModel):
    action: str = Field(description="'APPROVE' or 'REJECT'")
    faculty_notes: Optional[str] = None


# --- System Settings Schema ---
class SystemSettingsModel(BaseModel):
    recognition_threshold: float = 0.982
    liveness_threshold: float = 0.58
    late_grace_minutes: int = 10
    email_alerts_enabled: bool = False
    low_attendance_threshold_pct: float = 75.0
    require_liveness: bool = True
    geofencing_enabled: bool = True
    allow_manual_override: bool = True


# --- Audit Log Schema ---
class AuditLogResponse(BaseModel):
    id: str
    user_id: Optional[str] = None
    user_email: Optional[str] = None
    user_role: Optional[str] = None
    action: str
    entity_type: str
    entity_id: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)
    ip_address: Optional[str] = None
    timestamp: float
