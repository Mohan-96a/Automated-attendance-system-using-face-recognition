"""
Authentication and User Service for SmartAttend AI.
"""

import time
import uuid
from typing import Dict, Any, Optional, List
from fastapi import HTTPException, status

from backend.app.database.mongodb import db_manager
from backend.app.utils.security import hash_password, verify_password, create_access_token
from backend.app.services.audit_service import AuditService


class AuthService:
    @staticmethod
    async def authenticate_user(email: str, password: str, ip_address: Optional[str] = None) -> Dict[str, Any]:
        """Validates credentials and generates JWT access token."""
        user_col = db_manager.get_collection("users")
        user = await user_col.find_one({"email": email.lower().strip()})

        if not user or not verify_password(password, user.get("password_hash", "")):
            await AuditService.log_action(
                action="LOGIN_FAILED",
                entity_type="USER",
                user_email=email,
                details={"reason": "Invalid credentials"},
                ip_address=ip_address,
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
            )

        if not user.get("is_active", True):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Your account has been deactivated. Please contact an administrator.",
            )

        token_data = {
            "sub": user["id"],
            "email": user["email"],
            "role": user["role"],
            "full_name": user["full_name"],
        }
        access_token = create_access_token(token_data)

        await AuditService.log_action(
            action="LOGIN_SUCCESS",
            entity_type="USER",
            user_id=user["id"],
            user_email=user["email"],
            user_role=user["role"],
            ip_address=ip_address,
        )

        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user_id": user["id"],
            "email": user["email"],
            "full_name": user["full_name"],
            "role": user["role"],
        }

    @staticmethod
    async def register_user(
        email: str,
        password: str,
        full_name: str,
        role: str = "student",
        admin_user: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Registers a new user account."""
        user_col = db_manager.get_collection("users")
        existing = await user_col.find_one({"email": email.lower().strip()})
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="An account with this email address already exists.",
            )

        new_user = {
            "id": f"USR-{uuid.uuid4().hex[:10]}",
            "email": email.lower().strip(),
            "password_hash": hash_password(password),
            "full_name": full_name.strip(),
            "role": role,
            "is_active": True,
            "created_at": time.time(),
        }
        await user_col.insert_one(new_user)

        await AuditService.log_action(
            action="USER_REGISTERED",
            entity_type="USER",
            entity_id=new_user["id"],
            user_id=admin_user.get("id") if admin_user else new_user["id"],
            user_email=admin_user.get("email") if admin_user else email,
            details={"assigned_role": role},
        )
        return new_user
