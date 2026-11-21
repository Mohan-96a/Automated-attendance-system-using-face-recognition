"""
Faculty and Academic Management Services for SmartAttend AI.
"""

import time
import uuid
from typing import Dict, Any, List, Optional
from fastapi import HTTPException, status

from backend.app.database.mongodb import db_manager
from backend.app.services.audit_service import AuditService


class FacultyService:
    @staticmethod
    async def get_all_faculty(department_id: Optional[str] = None) -> List[Dict[str, Any]]:
        fac_col = db_manager.get_collection("faculty")
        query = {"department_id": department_id} if department_id else {}
        return await fac_col.find(query)

    @staticmethod
    async def get_faculty_by_id(faculty_id: str) -> Optional[Dict[str, Any]]:
        fac_col = db_manager.get_collection("faculty")
        faculty = await fac_col.find_one({"id": faculty_id})
        if not faculty:
            faculty = await fac_col.find_one({"user_id": faculty_id})
        return faculty

    @staticmethod
    async def create_faculty(data: Dict[str, Any], actor_user: Dict[str, Any]) -> Dict[str, Any]:
        fac_col = db_manager.get_collection("faculty")
        existing = await fac_col.find_one({"employee_id": data["employee_id"]})
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Faculty with Employee ID '{data['employee_id']}' already exists.",
            )

        new_faculty = {
            "id": f"FAC-{uuid.uuid4().hex[:8]}",
            "employee_id": data["employee_id"],
            "full_name": data["full_name"],
            "email": data["email"],
            "department_id": data["department_id"],
            "designation": data.get("designation", "Assistant Professor"),
            "phone": data.get("phone"),
            "assigned_subjects": data.get("assigned_subjects", []),
            "assigned_classes": data.get("assigned_classes", []),
            "created_at": time.time(),
        }
        await fac_col.insert_one(new_faculty)

        await AuditService.log_action(
            action="FACULTY_CREATED",
            entity_type="FACULTY",
            entity_id=new_faculty["id"],
            user_id=actor_user.get("id"),
            user_email=actor_user.get("email"),
            details={"employee_id": data["employee_id"], "name": data["full_name"]},
        )
        return new_faculty

    @staticmethod
    async def update_assignments(
        faculty_id: str,
        assigned_subjects: List[str],
        assigned_classes: List[str],
        actor_user: Dict[str, Any],
    ) -> Dict[str, Any]:
        fac_col = db_manager.get_collection("faculty")
        fac = await FacultyService.get_faculty_by_id(faculty_id)
        if not fac:
            raise HTTPException(status_code=404, detail="Faculty not found")

        await fac_col.update_one(
            {"id": fac["id"]},
            {"$set": {"assigned_subjects": assigned_subjects, "assigned_classes": assigned_classes}},
        )
        await AuditService.log_action(
            action="FACULTY_ASSIGNMENTS_UPDATED",
            entity_type="FACULTY",
            entity_id=fac["id"],
            user_id=actor_user.get("id"),
            user_email=actor_user.get("email"),
        )
        return await FacultyService.get_faculty_by_id(fac["id"])
