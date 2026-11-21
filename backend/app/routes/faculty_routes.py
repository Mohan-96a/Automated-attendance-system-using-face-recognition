"""
Faculty Management API Routes.
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status

from backend.app.schemas.student_schemas import (
    FacultyCreateRequest,
    FacultyUpdateRequest,
    FacultyResponse,
)
from backend.app.services.faculty_service import FacultyService
from backend.app.middleware.auth_middleware import get_current_user, RoleChecker

router = APIRouter(prefix="/faculty", tags=["Faculty"])


@router.get("", response_model=List[FacultyResponse])
async def list_faculty(
    department_id: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(RoleChecker(["admin", "faculty"])),
):
    """Lists faculty profiles."""
    return await FacultyService.get_all_faculty(department_id)


@router.get("/{id}", response_model=FacultyResponse)
async def get_faculty(
    id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Retrieves faculty profile details."""
    fac = await FacultyService.get_faculty_by_id(id)
    if not fac:
        raise HTTPException(status_code=404, detail="Faculty not found")
    return fac


@router.post("", response_model=FacultyResponse, status_code=status.HTTP_201_CREATED)
async def create_faculty(
    req: FacultyCreateRequest,
    current_user: Dict[str, Any] = Depends(RoleChecker(["admin"])),
):
    """Creates a new faculty record (Admin only)."""
    return await FacultyService.create_faculty(req.model_dump(), current_user)


@router.put("/{id}/assignments", response_model=FacultyResponse)
async def update_assignments(
    id: str,
    assigned_subjects: List[str],
    assigned_classes: List[str],
    current_user: Dict[str, Any] = Depends(RoleChecker(["admin"])),
):
    """Assigns subjects and classes to faculty."""
    return await FacultyService.update_assignments(id, assigned_subjects, assigned_classes, current_user)
