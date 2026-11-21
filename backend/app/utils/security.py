"""
Security Utilities: Password Hashing and JWT Token Management.
Uses standard PBKDF2-HMAC-SHA256 with random salt (zero external compilation dependencies)
and PyJWT for standard JWT token signing.
"""

import os
import time
import base64
import hashlib
import jwt
from typing import Dict, Any, Optional

from backend.app.config import settings


def hash_password(password: str) -> str:
    """Hashes a plaintext password with PBKDF2-HMAC-SHA256 and a random 16-byte salt."""
    salt = os.urandom(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
    # Store format: salt:key as hex
    return f"{salt.hex()}:{key.hex()}"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies plaintext password against stored salt:key hash."""
    try:
        salt_hex, key_hex = hashed_password.split(":")
        salt = bytes.fromhex(salt_hex)
        expected_key = bytes.fromhex(key_hex)
        key = hashlib.pbkdf2_hmac("sha256", plain_password.encode("utf-8"), salt, 100_000)
        return hashlib.sha256(key).digest() == hashlib.sha256(expected_key).digest()
    except Exception:
        return False


def create_access_token(data: Dict[str, Any], expires_delta: Optional[float] = None) -> str:
    """Encodes JWT access token with role, user_id, and expiration."""
    to_encode = data.copy()
    now = time.time()
    expire = now + (expires_delta if expires_delta else (settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60))
    to_encode.update({"exp": int(expire), "iat": int(now)})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    """Decodes and validates JWT token."""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return payload
    except jwt.PyJWTError:
        return None
