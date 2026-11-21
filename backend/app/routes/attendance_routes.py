"""
Attendance Management and Real-Time Face Recognition API Routes.
"""

import time
import uuid
import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status

from backend.app.schemas.attendance_schemas import (
    CreateSessionRequest,
    AttendanceSessionResponse,
    AttendanceRecordResponse,
    ProcessFrameRequest,
    ProcessFrameResponse,
    ManualAttendanceRequest,
    CreateCorrectionRequest,
    CorrectionRequestResponse,
    ReviewCorrectionRequest,
)
from backend.app.services.attendance_service import AttendanceService
from backend.app.services.audit_service import AuditService
from backend.app.middleware.auth_middleware import get_current_user, RoleChecker
from backend.app.database.mongodb import db_manager

logger = logging.getLogger("SmartAttend.AttendanceRoutes")
router = APIRouter(prefix="/attendance", tags=["Attendance"])


# --- Session Routes ---
@router.post("/sessions", response_model=AttendanceSessionResponse, status_code=status.HTTP_201_CREATED)
async def create_attendance_session(
    req: CreateSessionRequest,
    current_user: Dict[str, Any] = Depends(RoleChecker(["admin", "faculty"])),
):
    """Creates a new attendance session for a class (Faculty & Admin)."""
    return await AttendanceService.create_session(req.model_dump(), current_user)


@router.get("/sessions/active", response_model=List[AttendanceSessionResponse])
async def list_active_sessions(
    faculty_id: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Retrieves all currently active attendance sessions."""
    return await AttendanceService.get_active_sessions(faculty_id)


@router.get("/sessions/{id}", response_model=AttendanceSessionResponse)
async def get_session(id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    sess = await AttendanceService.get_session_by_id(id)
    if not sess:
        raise HTTPException(status_code=404, detail="Attendance session not found")
    return sess


@router.put("/sessions/{id}/status", response_model=AttendanceSessionResponse)
async def update_session_status(
    id: str,
    status: str,  # "ACTIVE", "PAUSED", "COMPLETED"
    current_user: Dict[str, Any] = Depends(RoleChecker(["admin", "faculty"])),
):
    """Starts, pauses, or terminates an attendance session."""
    return await AttendanceService.update_session_status(id, status, current_user)


# --- Live AI Frame Inference ---
@router.post("/sessions/{id}/process-frame", response_model=ProcessFrameResponse)
async def process_frame(
    id: str,
    req: ProcessFrameRequest,
    current_user: Dict[str, Any] = Depends(RoleChecker(["admin", "faculty"])),
):
    """
    Real-time AI camera frame processing endpoint:
    - Runs multi-face detection, distance estimation, face tracking, liveness verification, and ArcFace recognition
    - Automatically marks attendance with duplicate prevention and late arrival detection
    - Returns annotated frame with bounding boxes and recognized identities.
    """
    return await AttendanceService.process_live_frame(
        session_id=id,
        frame_base64=req.frame_base64,
        user_lat=req.user_latitude,
        user_lon=req.user_longitude,
    )


# --- Records and Manual Override ---
@router.post("/sessions/{id}/manual")
async def manual_mark_attendance(
    id: str,
    req: ManualAttendanceRequest,
    current_user: Dict[str, Any] = Depends(RoleChecker(["admin", "faculty"])),
):
    """Allows faculty to manually mark or adjust attendance status."""
    return await AttendanceService.manual_mark_attendance(
        session_id=id,
        student_id=req.student_id,
        new_status=req.status,
        is_late=req.is_late,
        notes=req.notes,
        actor_user=current_user,
    )


@router.get("/sessions/{id}/records", response_model=List[AttendanceRecordResponse])
async def get_session_records(
    id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Returns all attendance records for a given session."""
    return await AttendanceService.get_session_records(id)


@router.get("/student/{id}/records", response_model=List[AttendanceRecordResponse])
async def get_student_records(
    id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Returns all historical attendance records for a student."""
    return await AttendanceService.get_student_records(id)


# --- Attendance Correction Workflow ---
@router.post("/corrections", response_model=CorrectionRequestResponse, status_code=status.HTTP_201_CREATED)
async def submit_correction_request(
    req: CreateCorrectionRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Students can request an attendance correction for review by faculty."""
    corr_col = db_manager.get_collection("correction_requests")
    stu_col = db_manager.get_collection("students")
    sess_col = db_manager.get_collection("attendance_sessions")

    student = await stu_col.find_one({"id": req.student_id})
    session = await sess_col.find_one({"id": req.session_id})

    doc = {
        "id": f"CORR-{uuid.uuid4().hex[:8]}",
        "session_id": req.session_id,
        "student_id": req.student_id,
        "student_name": student.get("full_name", "Student") if student else "Student",
        "date": session.get("date", "") if session else "",
        "reason": req.reason,
        "requested_status": req.requested_status,
        "status": "PENDING",
        "faculty_notes": None,
        "created_at": time.time(),
    }
    await corr_col.insert_one(doc)
    await AuditService.log_action("CORRECTION_REQUESTED", "ATTENDANCE_CORRECTION", entity_id=doc["id"])
    return doc


@router.get("/corrections", response_model=List[CorrectionRequestResponse])
async def list_correction_requests(
    status: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(RoleChecker(["admin", "faculty"])),
):
    """Lists correction requests for faculty/admin review."""
    corr_col = db_manager.get_collection("correction_requests")
    query = {"status": status} if status else {}
    return await corr_col.find(query)


@router.put("/corrections/{id}/review", response_model=CorrectionRequestResponse)
async def review_correction_request(
    id: str,
    req: ReviewCorrectionRequest,
    current_user: Dict[str, Any] = Depends(RoleChecker(["admin", "faculty"])),
):
    """Approves or rejects a student's attendance correction request."""
    corr_col = db_manager.get_collection("correction_requests")
    corr = await corr_col.find_one({"id": id})
    if not corr:
        raise HTTPException(status_code=404, detail="Correction request not found")

    new_status = "APPROVED" if req.action.upper() == "APPROVE" else "REJECTED"
    await corr_col.update_one(
        {"id": id},
        {"$set": {"status": new_status, "faculty_notes": req.faculty_notes}},
    )

    if new_status == "APPROVED":
        # Update attendance record
        await AttendanceService.manual_mark_attendance(
            session_id=corr["session_id"],
            student_id=corr["student_id"],
            new_status=corr["requested_status"],
            is_late=False,
            notes=f"Approved correction: {req.faculty_notes or ''}",
            actor_user=current_user,
        )

    await AuditService.log_action(
        action=f"CORRECTION_{new_status}",
        entity_type="ATTENDANCE_CORRECTION",
        entity_id=id,
        user_id=current_user.get("id"),
        user_email=current_user.get("email"),
    )
    return await corr_col.find_one({"id": id})
