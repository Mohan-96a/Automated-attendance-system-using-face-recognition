"""
Pydantic Schemas for Authentication and User Management.
"""

from typing import Optional, List
from pydantic import BaseModel, Field


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    email: str
    full_name: str
    role: str


class LoginRequest(BaseModel):
    email: str = Field(description="User email address")
    password: str


class RegisterRequest(BaseModel):
    email: str = Field(description="User email address")
    password: str = Field(min_length=6)
    full_name: str
    role: str = "student"  # "admin", "faculty", "student"


class PasswordResetRequest(BaseModel):
    email: str = Field(description="User email address")


class PasswordResetConfirm(BaseModel):
    token: str
    new_password: str = Field(min_length=6)


class UserProfileResponse(BaseModel):
    id: str
    email: str
    full_name: str
    role: str
    is_active: bool
    created_at: float


class UpdateUserRequest(BaseModel):
    full_name: Optional[str] = None
    role: Optional[str] = None
    is_active: Optional[bool] = None
