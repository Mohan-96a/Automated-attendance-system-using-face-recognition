"""
Attendance Management and Real-Time AI Inference Service for SmartAttend AI.
"""

import time
import uuid
import base64
import math
import datetime
import logging
import cv2
import numpy as np
from typing import Dict, Any, List, Optional
from fastapi import HTTPException, status

from backend.app.database.mongodb import db_manager
from backend.app.services.audit_service import AuditService
from backend.app.services.student_service import get_shared_ai_engine
from backend.app.config import settings

logger = logging.getLogger("SmartAttend.AttendanceService")


def calculate_haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates distance between two GPS coordinates in meters."""
    R = 6371000.0  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * (math.sin(delta_lambda / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


class AttendanceService:
    @staticmethod
    async def create_session(data: Dict[str, Any], actor_user: Dict[str, Any]) -> Dict[str, Any]:
        """Creates a new attendance session for a class/subject."""
        sess_col = db_manager.get_collection("attendance_sessions")
        sub_col = db_manager.get_collection("subjects")
        cr_col = db_manager.get_collection("classrooms")
        fac_col = db_manager.get_collection("faculty")
        stu_col = db_manager.get_collection("students")

        subject = await sub_col.find_one({"id": data["subject_id"]})
        classroom = await cr_col.find_one({"id": data["classroom_id"]})
        faculty = await fac_col.find_one({"id": data["faculty_id"]})

        date_str = data.get("date") or datetime.date.today().strftime("%Y-%m-%d")
        sess_id = f"SESS-{uuid.uuid4().hex[:8]}"

        # Count enrolled students in this department
        dept_id = subject.get("department_id") if subject else None
        stu_query = {"department_id": dept_id} if dept_id else {}
        total_enrolled = await stu_col.count_documents(stu_query)

        session_doc = {
            "id": sess_id,
            "subject_id": data["subject_id"],
            "subject_name": subject.get("name", "Subject") if subject else data["subject_id"],
            "classroom_id": data["classroom_id"],
            "classroom_name": classroom.get("name", "Classroom") if classroom else data["classroom_id"],
            "faculty_id": data["faculty_id"],
            "faculty_name": faculty.get("full_name", "Faculty") if faculty else data["faculty_id"],
            "date": date_str,
            "start_time": data.get("start_time", "09:00:00"),
            "end_time": data.get("end_time", "10:00:00"),
            "status": "ACTIVE",
            "late_grace_minutes": data.get("late_grace_minutes", 10),
            "total_enrolled": total_enrolled or 5,
            "present_count": 0,
            "absent_count": total_enrolled or 5,
            "created_at": time.time(),
        }
        await sess_col.insert_one(session_doc)

        # Pre-seed ABSENT attendance records for enrolled students
        rec_col = db_manager.get_collection("attendance_records")
        enrolled_students = await stu_col.find(stu_query)
        for stu in enrolled_students:
            rec_doc = {
                "id": f"REC-{sess_id}-{stu['id']}",
                "session_id": sess_id,
                "student_id": stu["id"],
                "student_name": stu["full_name"],
                "subject_id": data["subject_id"],
                "date": date_str,
                "timestamp": None,
                "status": "ABSENT",
                "is_late": False,
                "verification_method": "NONE",
                "confidence_score": 0.0,
                "distance_cm": 0,
            }
            await rec_col.insert_one(rec_doc)

        await AuditService.log_action(
            action="ATTENDANCE_SESSION_CREATED",
            entity_type="ATTENDANCE_SESSION",
            entity_id=sess_id,
            user_id=actor_user.get("id"),
            user_email=actor_user.get("email"),
        )
        return session_doc

    @staticmethod
    async def get_active_sessions(faculty_id: Optional[str] = None) -> List[Dict[str, Any]]:
        sess_col = db_manager.get_collection("attendance_sessions")
        query: Dict[str, Any] = {"status": "ACTIVE"}
        if faculty_id:
            query["faculty_id"] = faculty_id
        return await sess_col.find(query)

    @staticmethod
    async def get_session_by_id(session_id: str) -> Optional[Dict[str, Any]]:
        sess_col = db_manager.get_collection("attendance_sessions")
        return await sess_col.find_one({"id": session_id})

    @staticmethod
    async def update_session_status(session_id: str, new_status: str, actor_user: Dict[str, Any]) -> Dict[str, Any]:
        sess_col = db_manager.get_collection("attendance_sessions")
        await sess_col.update_one({"id": session_id}, {"$set": {"status": new_status}})
        await AuditService.log_action(
            action=f"SESSION_STATUS_{new_status}",
            entity_type="ATTENDANCE_SESSION",
            entity_id=session_id,
            user_id=actor_user.get("id"),
            user_email=actor_user.get("email"),
        )
        return await sess_col.find_one({"id": session_id})

    @staticmethod
    async def process_live_frame(
        session_id: str,
        frame_base64: str,
        user_lat: Optional[float] = None,
        user_lon: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Ingests a real-time webcam frame from the classroom camera or browser:
        1. Validates session status (must be ACTIVE)
        2. Validates classroom geofencing if coordinates are provided
        3. Invokes SmartAttendAIEngine to detect, track, verify liveness, and recognize faces
        4. Prevents duplicate marking for already present students
        5. Automatically detects late arrival based on session start time + grace period
        6. Updates attendance records and session counters
        7. Returns annotated overlay frame and event stream.
        """
        session = await AttendanceService.get_session_by_id(session_id)
        if not session or session.get("status") != "ACTIVE":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Attendance session is not active.",
            )

        # Geofencing check
        if user_lat is not None and user_lon is not None and settings.DEFAULT_LATE_GRACE_MINUTES:
            cr_col = db_manager.get_collection("classrooms")
            classroom = await cr_col.find_one({"id": session.get("classroom_id")})
            if classroom and classroom.get("geofence"):
                gf = classroom["geofence"]
                dist_m = calculate_haversine_distance(
                    user_lat, user_lon, gf["latitude"], gf["longitude"]
                )
                if dist_m > gf.get("radius_meters", 50.0):
                    logger.warning("Geofence violation: %.1fm > %.1fm", dist_m, gf.get("radius_meters", 50.0))

        # Decode base64 frame
        try:
            if "," in frame_base64:
                frame_base64 = frame_base64.split(",")[1]
            img_bytes = base64.b64decode(frame_base64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("Could not decode frame image")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Frame decode error: {e}")

        # Sync known faces into AI Engine if not already cached
        ai_engine = get_shared_ai_engine()
        if len(ai_engine.recognizer.known_profiles) == 0:
            emb_col = db_manager.get_collection("face_embeddings")
            stu_col = db_manager.get_collection("students")
            all_embs = await emb_col.find()
            for emb in all_embs:
                stu = await stu_col.find_one({"id": emb["student_id"]})
                if stu:
                    ai_engine.recognizer.load_encrypted_profile(
                        student_id=stu["id"],
                        name=stu["full_name"],
                        encrypted_embedding=emb["encrypted_embedding"],
                    )

        # Process frame through AI Engine
        res = ai_engine.process_frame(frame, draw_overlay=True)

        rec_col = db_manager.get_collection("attendance_records")
        sess_col = db_manager.get_collection("attendance_sessions")

        recognized_students_payload = []
        now_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Check late status: compare current time with start_time
        start_time_str = session.get("start_time", "09:00:00")
        grace_mins = session.get("late_grace_minutes", 10)
        try:
            now_dt = datetime.datetime.now()
            start_parts = [int(p) for p in start_time_str.split(":")]
            start_dt = now_dt.replace(hour=start_parts[0], minute=start_parts[1], second=start_parts[2])
            is_late_flag = (now_dt - start_dt).total_seconds() > (grace_mins * 60)
        except Exception:
            is_late_flag = False

        for face in res.processed_faces:
            if face.is_ready_for_attendance and face.student_id:
                # Check duplicate in this session
                existing_rec = await rec_col.find_one({
                    "session_id": session_id,
                    "student_id": face.student_id,
                })

                if not existing_rec or existing_rec.get("status") != "PRESENT":
                    # Mark attendance!
                    rec_update = {
                        "status": "PRESENT",
                        "timestamp": now_ts,
                        "is_late": is_late_flag,
                        "verification_method": "FACE_RECOGNITION",
                        "confidence_score": face.confidence_percentage,
                        "distance_cm": face.distance_cm,
                    }
                    if existing_rec:
                        await rec_col.update_one({"id": existing_rec["id"]}, {"$set": rec_update})
                    else:
                        rec_doc = {
                            "id": f"REC-{session_id}-{face.student_id}",
                            "session_id": session_id,
                            "student_id": face.student_id,
                            "student_name": face.name,
                            "subject_id": session.get("subject_id"),
                            "date": session.get("date"),
                            **rec_update,
                        }
                        await rec_col.insert_one(rec_doc)

                    # Update session counters
                    present_cnt = await rec_col.count_documents({"session_id": session_id, "status": "PRESENT"})
                    total_cnt = session.get("total_enrolled", 5)
                    await sess_col.update_one(
                        {"id": session_id},
                        {"$set": {"present_count": present_cnt, "absent_count": max(0, total_cnt - present_cnt)}},
                    )
                    logger.info("Marked attendance for %s in session %s (Late: %s)", face.name, session_id, is_late_flag)

                recognized_students_payload.append({
                    "student_id": face.student_id,
                    "name": face.name,
                    "confidence_percentage": face.confidence_percentage,
                    "distance_cm": face.distance_cm,
                    "distance_status": face.distance_status,
                    "is_live": face.is_live,
                    "is_late": is_late_flag,
                })

        # Encode annotated frame back to JPEG base64 for live UI streaming
        annotated_b64 = None
        if res.annotated_frame is not None:
            _, buffer = cv2.imencode(".jpg", res.annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            annotated_b64 = base64.b64encode(buffer).decode("utf-8")

        return {
            "success": True,
            "faces_detected": res.total_faces_detected,
            "multiple_faces_alert": res.multiple_faces_detected,
            "annotated_frame_base64": annotated_b64,
            "recognized_students": recognized_students_payload,
            "processing_time_ms": res.processing_time_ms,
            "fps": res.fps,
        }

    @staticmethod
    async def manual_mark_attendance(
        session_id: str,
        student_id: str,
        new_status: str,
        is_late: bool,
        notes: Optional[str],
        actor_user: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Allows faculty to manually override attendance status with audit tracking."""
        rec_col = db_manager.get_collection("attendance_records")
        sess_col = db_manager.get_collection("attendance_sessions")
        stu_col = db_manager.get_collection("students")

        record = await rec_col.find_one({"session_id": session_id, "student_id": student_id})
        student = await stu_col.find_one({"id": student_id})
        now_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        update_fields = {
            "status": new_status,
            "is_late": is_late,
            "verification_method": "MANUAL_FACULTY_OVERRIDE",
            "timestamp": now_ts if new_status == "PRESENT" else None,
            "override_by": actor_user.get("email"),
            "override_notes": notes,
        }

        if record:
            await rec_col.update_one({"id": record["id"]}, {"$set": update_fields})
        else:
            doc = {
                "id": f"REC-{session_id}-{student_id}",
                "session_id": session_id,
                "student_id": student_id,
                "student_name": student.get("full_name", "Student") if student else "Student",
                "date": datetime.date.today().strftime("%Y-%m-%d"),
                **update_fields,
            }
            await rec_col.insert_one(doc)

        # Update session counters
        present_cnt = await rec_col.count_documents({"session_id": session_id, "status": "PRESENT"})
        sess = await sess_col.find_one({"id": session_id})
        total_cnt = sess.get("total_enrolled", 5) if sess else 5
        await sess_col.update_one(
            {"id": session_id},
            {"$set": {"present_count": present_cnt, "absent_count": max(0, total_cnt - present_cnt)}},
        )

        await AuditService.log_action(
            action="MANUAL_ATTENDANCE_OVERRIDE",
            entity_type="ATTENDANCE_RECORD",
            entity_id=student_id,
            user_id=actor_user.get("id"),
            user_email=actor_user.get("email"),
            details={"status": new_status, "is_late": is_late, "notes": notes},
        )
        return {"success": True, "message": f"Attendance updated to {new_status}"}

    @staticmethod
    async def get_session_records(session_id: str) -> List[Dict[str, Any]]:
        rec_col = db_manager.get_collection("attendance_records")
        return await rec_col.find({"session_id": session_id})

    @staticmethod
    async def get_student_records(student_id: str) -> List[Dict[str, Any]]:
        rec_col = db_manager.get_collection("attendance_records")
        return await rec_col.find({"student_id": student_id})
