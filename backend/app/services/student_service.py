"""
Student Management and Biometric Enrollment Service for SmartAttend AI.
"""

import time
import uuid
import base64
import logging
import cv2
import numpy as np
import pandas as pd
from io import BytesIO
from typing import Dict, Any, List, Optional
from fastapi import HTTPException, status

from backend.app.database.mongodb import db_manager
from backend.app.services.audit_service import AuditService
from ai_engine.engine import SmartAttendAIEngine

logger = logging.getLogger("SmartAttend.StudentService")

# Global singleton AI Engine shared by backend services
_global_ai_engine: Optional[SmartAttendAIEngine] = None


def get_shared_ai_engine() -> SmartAttendAIEngine:
    global _global_ai_engine
    if _global_ai_engine is None:
        _global_ai_engine = SmartAttendAIEngine()
    return _global_ai_engine


class StudentService:
    @staticmethod
    async def get_all_students(
        department_id: Optional[str] = None,
        year: Optional[int] = None,
        search_query: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieves list of students with optional department/year filtering and text search."""
        stu_col = db_manager.get_collection("students")
        query: Dict[str, Any] = {}
        if department_id:
            query["department_id"] = department_id
        if year:
            query["year"] = year

        students = await stu_col.find(query)
        if search_query:
            sq = search_query.lower()
            students = [
                s for s in students
                if sq in s.get("full_name", "").lower() or sq in s.get("student_id", "").lower()
            ]
        return students

    @staticmethod
    async def get_student_by_id(student_internal_id: str) -> Optional[Dict[str, Any]]:
        stu_col = db_manager.get_collection("students")
        student = await stu_col.find_one({"id": student_internal_id})
        if not student:
            # Check by roll number student_id
            student = await stu_col.find_one({"student_id": student_internal_id})
        return student

    @staticmethod
    async def create_student(data: Dict[str, Any], actor_user: Dict[str, Any]) -> Dict[str, Any]:
        stu_col = db_manager.get_collection("students")
        existing = await stu_col.find_one({"student_id": data["student_id"]})
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Student ID '{data['student_id']}' is already registered.",
            )

        new_student = {
            "id": f"STU-{uuid.uuid4().hex[:10]}",
            "student_id": data["student_id"],
            "full_name": data["full_name"],
            "email": data["email"],
            "department_id": data["department_id"],
            "year": data.get("year", 1),
            "section": data.get("section", "A"),
            "phone": data.get("phone"),
            "face_enrolled": False,
            "overall_attendance_pct": 0.0,
            "risk_status": "LOW_RISK",
            "created_at": time.time(),
        }
        await stu_col.insert_one(new_student)

        await AuditService.log_action(
            action="STUDENT_CREATED",
            entity_type="STUDENT",
            entity_id=new_student["id"],
            user_id=actor_user.get("id"),
            user_email=actor_user.get("email"),
            details={"student_id": data["student_id"], "name": data["full_name"]},
        )
        return new_student

    @staticmethod
    async def enroll_face(
        student_id: str,
        image_base64: str,
        actor_user: Dict[str, Any],
        enforce_liveness: bool = True,
    ) -> Dict[str, Any]:
        """
        Processes facial biometric onboarding for a student:
        decodes base64 frame, validates quality and liveness, extracts 512-d ArcFace embedding,
        encrypts template with AES/Fernet, and stores in face_embeddings collection.
        """
        student = await StudentService.get_student_by_id(student_id)
        if not student:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Student record not found",
            )

        # Decode base64 image
        try:
            if "," in image_base64:
                image_base64 = image_base64.split(",")[1]
            img_bytes = base64.b64decode(image_base64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("Could not decode image")
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid image format: {e}",
            )

        ai_engine = get_shared_ai_engine()
        enroll_res = ai_engine.enrollment.enroll_from_image(
            student_id=student["id"],
            name=student["full_name"],
            image_or_path=frame,
            enforce_liveness=enforce_liveness,
            metadata={"student_code": student["student_id"], "department": student["department_id"]},
        )

        if not enroll_res.success:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "message": enroll_res.message,
                    "errors": enroll_res.validation_errors,
                    "quality_score": enroll_res.quality_score,
                    "liveness_score": enroll_res.liveness_score,
                },
            )

        # Store encrypted biometric template
        emb_col = db_manager.get_collection("face_embeddings")
        await emb_col.delete_one({"student_id": student["id"]})
        emb_doc = {
            "id": f"EMB-{uuid.uuid4().hex[:10]}",
            "student_id": student["id"],
            "student_code": student["student_id"],
            "encrypted_embedding": enroll_res.encrypted_embedding,
            "quality_score": enroll_res.quality_score,
            "thumbnail_base64": enroll_res.thumbnail_base64,
            "created_at": time.time(),
        }
        await emb_col.insert_one(emb_doc)

        # Update student record
        stu_col = db_manager.get_collection("students")
        await stu_col.update_one({"id": student["id"]}, {"$set": {"face_enrolled": True}})

        await AuditService.log_action(
            action="BIOMETRIC_ENROLLED",
            entity_type="BIOMETRIC",
            entity_id=student["id"],
            user_id=actor_user.get("id"),
            user_email=actor_user.get("email"),
            details={"student_name": student["full_name"]},
        )

        return {
            "success": True,
            "message": "Facial biometrics enrolled and encrypted successfully.",
            "student_id": student["student_id"],
            "face_enrolled": True,
            "quality_score": enroll_res.quality_score,
            "liveness_score": enroll_res.liveness_score,
            "thumbnail_base64": enroll_res.thumbnail_base64,
        }

    @staticmethod
    async def delete_biometrics(student_id: str, actor_user: Dict[str, Any]) -> Dict[str, Any]:
        """Deletes student facial biometric template (GDPR right to be forgotten)."""
        student = await StudentService.get_student_by_id(student_id)
        if not student:
            raise HTTPException(status_code=404, detail="Student not found")

        emb_col = db_manager.get_collection("face_embeddings")
        await emb_col.delete_one({"student_id": student["id"]})

        stu_col = db_manager.get_collection("students")
        await stu_col.update_one({"id": student["id"]}, {"$set": {"face_enrolled": False}})

        # Remove from running AI engine memory
        ai_engine = get_shared_ai_engine()
        ai_engine.recognizer.unregister_face(student["id"])

        await AuditService.log_action(
            action="BIOMETRIC_DELETED",
            entity_type="BIOMETRIC",
            entity_id=student["id"],
            user_id=actor_user.get("id"),
            user_email=actor_user.get("email"),
        )
        return {"success": True, "message": "Biometric templates securely erased."}

    @staticmethod
    async def bulk_import_students(file_bytes: bytes, filename: str, actor_user: Dict[str, Any]) -> Dict[str, Any]:
        """Imports students from CSV or Excel file."""
        try:
            if filename.endswith(".csv"):
                df = pd.read_csv(BytesIO(file_bytes))
            elif filename.endswith((".xlsx", ".xls")):
                df = pd.read_excel(BytesIO(file_bytes))
            else:
                raise ValueError("Unsupported file format. Please upload .csv or .xlsx")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read spreadsheet: {e}")

        required_cols = {"student_id", "full_name", "email", "department_id"}
        if not required_cols.issubset(df.columns):
            raise HTTPException(
                status_code=400,
                detail=f"Spreadsheet must contain columns: {', '.join(required_cols)}",
            )

        imported = 0
        stu_col = db_manager.get_collection("students")

        for _, row in df.iterrows():
            s_id = str(row["student_id"]).strip()
            existing = await stu_col.find_one({"student_id": s_id})
            if existing:
                continue

            doc = {
                "id": f"STU-{uuid.uuid4().hex[:10]}",
                "student_id": s_id,
                "full_name": str(row["full_name"]).strip(),
                "email": str(row["email"]).strip(),
                "department_id": str(row["department_id"]).strip(),
                "year": int(row.get("year", 1)),
                "section": str(row.get("section", "A")).strip(),
                "phone": str(row.get("phone", "")) if pd.notna(row.get("phone")) else None,
                "face_enrolled": False,
                "overall_attendance_pct": 0.0,
                "risk_status": "LOW_RISK",
                "created_at": time.time(),
            }
            await stu_col.insert_one(doc)
            imported += 1

        await AuditService.log_action(
            action="BULK_STUDENT_IMPORT",
            entity_type="STUDENT",
            user_id=actor_user.get("id"),
            user_email=actor_user.get("email"),
            details={"total_imported": imported, "filename": filename},
        )
        return {"success": True, "total_imported": imported}
