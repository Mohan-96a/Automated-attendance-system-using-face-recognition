"""
Academic Management API Routes: Departments, Classrooms, Subjects, Timetables.
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status

from backend.app.schemas.academic_schemas import (
    DepartmentCreateRequest,
    DepartmentResponse,
    ClassroomCreateRequest,
    ClassroomResponse,
    SubjectCreateRequest,
    SubjectResponse,
    TimetableCreateRequest,
    TimetableResponse,
)
from backend.app.services.academic_service import AcademicService
from backend.app.middleware.auth_middleware import get_current_user, RoleChecker

router = APIRouter(prefix="/academics", tags=["Academics"])


# --- Departments ---
@router.get("/departments", response_model=List[DepartmentResponse])
async def list_departments(current_user: Dict[str, Any] = Depends(get_current_user)):
    return await AcademicService.get_departments()


@router.post("/departments", response_model=DepartmentResponse, status_code=status.HTTP_201_CREATED)
async def create_department(
    req: DepartmentCreateRequest,
    current_user: Dict[str, Any] = Depends(RoleChecker(["admin"])),
):
    return await AcademicService.create_department(req.model_dump(), current_user)


# --- Classrooms ---
@router.get("/classrooms", response_model=List[ClassroomResponse])
async def list_classrooms(current_user: Dict[str, Any] = Depends(get_current_user)):
    return await AcademicService.get_classrooms()


@router.post("/classrooms", response_model=ClassroomResponse, status_code=status.HTTP_201_CREATED)
async def create_classroom(
    req: ClassroomCreateRequest,
    current_user: Dict[str, Any] = Depends(RoleChecker(["admin"])),
):
    return await AcademicService.create_classroom(req.model_dump(), current_user)


# --- Subjects ---
@router.get("/subjects", response_model=List[SubjectResponse])
async def list_subjects(
    department_id: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    return await AcademicService.get_subjects(department_id)


@router.post("/subjects", response_model=SubjectResponse, status_code=status.HTTP_201_CREATED)
async def create_subject(
    req: SubjectCreateRequest,
    current_user: Dict[str, Any] = Depends(RoleChecker(["admin"])),
):
    return await AcademicService.create_subject(req.model_dump(), current_user)


# --- Timetables ---
@router.get("/timetables", response_model=List[TimetableResponse])
async def list_timetables(
    day_of_week: Optional[str] = None,
    faculty_id: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    return await AcademicService.get_timetables(day_of_week, faculty_id)


@router.post("/timetables", response_model=TimetableResponse, status_code=status.HTTP_201_CREATED)
async def create_timetable(
    req: TimetableCreateRequest,
    current_user: Dict[str, Any] = Depends(RoleChecker(["admin"])),
):
    return await AcademicService.create_timetable(req.model_dump(), current_user)
