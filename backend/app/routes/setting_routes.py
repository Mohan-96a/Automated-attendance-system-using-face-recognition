"""
System Settings and Audit Log API Routes.
"""

from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, status

from backend.app.schemas.attendance_schemas import SystemSettingsModel, AuditLogResponse
from backend.app.middleware.auth_middleware import get_current_user, RoleChecker
from backend.app.database.mongodb import db_manager
from backend.app.services.audit_service import AuditService

router = APIRouter(tags=["Settings & Audit"])


@router.get("/settings", response_model=SystemSettingsModel)
async def get_system_settings(current_user: Dict[str, Any] = Depends(RoleChecker(["admin"]))):
    """Retrieves system configuration (recognition threshold, liveness sensitivity, etc.)."""
    settings_col = db_manager.get_collection("system_settings")
    sett = await settings_col.find_one({"id": "SYS-SETTINGS-01"})
    if not sett:
        sett = SystemSettingsModel().model_dump()
        sett["id"] = "SYS-SETTINGS-01"
        await settings_col.insert_one(sett)
    return sett


@router.put("/settings", response_model=SystemSettingsModel)
async def update_system_settings(
    req: SystemSettingsModel,
    current_user: Dict[str, Any] = Depends(RoleChecker(["admin"])),
):
    """Updates global AI thresholds and attendance rules."""
    settings_col = db_manager.get_collection("system_settings")
    data = req.model_dump()
    await settings_col.update_one({"id": "SYS-SETTINGS-01"}, {"$set": data})
    await AuditService.log_action("SETTINGS_UPDATED", "SYSTEM_SETTINGS", user_id=current_user.get("id"))
    return await settings_col.find_one({"id": "SYS-SETTINGS-01"})


@router.get("/audit-logs", response_model=List[AuditLogResponse])
async def get_audit_logs(
    limit: int = 50,
    current_user: Dict[str, Any] = Depends(RoleChecker(["admin"])),
):
    """Retrieves immutable security and activity history logs."""
    audit_col = db_manager.get_collection("audit_logs")
    logs = await audit_col.find(limit=limit)
    # Sort reverse by timestamp
    logs.sort(key=lambda l: l.get("timestamp", 0), reverse=True)
    return logs
