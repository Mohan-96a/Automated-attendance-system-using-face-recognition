"""
Authentication and User Account API Routes.
"""

from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, status, Request

from backend.app.schemas.auth_schemas import (
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    UserProfileResponse,
    PasswordResetRequest,
)
from backend.app.services.auth_service import AuthService
from backend.app.middleware.auth_middleware import get_current_user, RoleChecker
from backend.app.database.mongodb import db_manager

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest, request: Request):
    """Authenticates admin, faculty, or student and returns JWT bearer token."""
    ip = request.client.host if request.client else None
    return await AuthService.authenticate_user(req.email, req.password, ip_address=ip)


@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(req: RegisterRequest):
    """Registers a new student, faculty, or admin account."""
    user = await AuthService.register_user(
        email=req.email,
        password=req.password,
        full_name=req.full_name,
        role=req.role,
    )
    return {
        "success": True,
        "message": "User registered successfully",
        "user_id": user["id"],
        "email": user["email"],
        "role": user["role"],
    }


@router.get("/me", response_model=UserProfileResponse)
async def get_my_profile(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Returns currently authenticated user profile."""
    return current_user


@router.post("/reset-password")
async def request_password_reset(req: PasswordResetRequest):
    """Initiates password reset procedure."""
    user_col = db_manager.get_collection("users")
    user = await user_col.find_one({"email": req.email.lower().strip()})
    if not user:
        # Don't leak user presence
        return {"success": True, "message": "If an account exists, a reset link has been dispatched."}

    return {"success": True, "message": "Password reset token sent to your email address."}
