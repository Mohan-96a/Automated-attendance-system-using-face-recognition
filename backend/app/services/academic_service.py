"""
Academic Services: Departments, Classrooms, Subjects, and Timetables.
"""

import time
import uuid
from typing import Dict, Any, List, Optional
from fastapi import HTTPException, status

from backend.app.database.mongodb import db_manager
from backend.app.services.audit_service import AuditService


class AcademicService:
    # --- Departments ---
    @staticmethod
    async def get_departments() -> List[Dict[str, Any]]:
        return await db_manager.get_collection("departments").find()

    @staticmethod
    async def create_department(data: Dict[str, Any], actor_user: Dict[str, Any]) -> Dict[str, Any]:
        dept_col = db_manager.get_collection("departments")
        existing = await dept_col.find_one({"id": data["id"]})
        if existing:
            raise HTTPException(status_code=400, detail="Department ID already exists")
        await dept_col.insert_one(data)
        await AuditService.log_action("DEPT_CREATED", "DEPARTMENT", entity_id=data["id"])
        return data

    # --- Classrooms & Geofencing ---
    @staticmethod
    async def get_classrooms() -> List[Dict[str, Any]]:
        return await db_manager.get_collection("classrooms").find()

    @staticmethod
    async def create_classroom(data: Dict[str, Any], actor_user: Dict[str, Any]) -> Dict[str, Any]:
        cr_col = db_manager.get_collection("classrooms")
        existing = await cr_col.find_one({"id": data["id"]})
        if existing:
            raise HTTPException(status_code=400, detail="Classroom ID already exists")
        await cr_col.insert_one(data)
        await AuditService.log_action("CLASSROOM_CREATED", "CLASSROOM", entity_id=data["id"])
        return data

    # --- Subjects ---
    @staticmethod
    async def get_subjects(department_id: Optional[str] = None) -> List[Dict[str, Any]]:
        query = {"department_id": department_id} if department_id else {}
        return await db_manager.get_collection("subjects").find(query)

    @staticmethod
    async def create_subject(data: Dict[str, Any], actor_user: Dict[str, Any]) -> Dict[str, Any]:
        sub_col = db_manager.get_collection("subjects")
        existing = await sub_col.find_one({"id": data["id"]})
        if existing:
            raise HTTPException(status_code=400, detail="Subject ID already exists")
        await sub_col.insert_one(data)
        await AuditService.log_action("SUBJECT_CREATED", "SUBJECT", entity_id=data["id"])
        return data

    # --- Timetables ---
    @staticmethod
    async def get_timetables(day_of_week: Optional[str] = None, faculty_id: Optional[str] = None) -> List[Dict[str, Any]]:
        query = {}
        if day_of_week:
            query["day_of_week"] = day_of_week
        if faculty_id:
            query["faculty_id"] = faculty_id
        return await db_manager.get_collection("timetables").find(query)

    @staticmethod
    async def create_timetable(data: Dict[str, Any], actor_user: Dict[str, Any]) -> Dict[str, Any]:
        tt_col = db_manager.get_collection("timetables")
        data["id"] = data.get("id", f"TT-{uuid.uuid4().hex[:8]}")
        await tt_col.insert_one(data)
        await AuditService.log_action("TIMETABLE_CREATED", "TIMETABLE", entity_id=data["id"])
        return data
