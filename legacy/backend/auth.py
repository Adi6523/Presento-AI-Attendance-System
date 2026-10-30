"""JWT authentication, password hashing, and role-based access control."""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Literal

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

from .database import connect

# ── Configuration ─────────────────────────────────────────────────────────────
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "presento-dev-secret-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "480"))  # 8 hours

Role = Literal["admin", "teacher", "student"]

# ── Crypto helpers ─────────────────────────────────────────────────────────────
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


# ── Token helpers ──────────────────────────────────────────────────────────────
def create_access_token(data: dict) -> str:
    payload = data.copy()
    payload["exp"] = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")


# ── FastAPI dependencies ───────────────────────────────────────────────────────
def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    """Returns the full user row for the authenticated JWT bearer."""
    payload = decode_token(token)
    user_id: int | None = payload.get("sub")
    if user_id is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")
    with connect() as db:
        user = db.execute(
            "SELECT id, username, email, role, student_id, active FROM users WHERE id=?",
            (int(user_id),)
        ).fetchone()
    if not user or not user["active"]:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found or deactivated")
    return dict(user)


def require_role(*roles: Role):
    """Dependency factory: raises 403 if the authenticated user's role is not in `roles`."""
    def _check(current_user: dict = Depends(get_current_user)):
        if current_user["role"] not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access restricted to roles: {', '.join(roles)}"
            )
        return current_user
    return _check


def authenticate_user(username: str, password: str) -> dict | None:
    """Returns the user dict if credentials are valid, otherwise None."""
    with connect() as db:
        user = db.execute(
            "SELECT id, username, email, password_hash, role, student_id, active FROM users WHERE username=? AND active=1",
            (username,)
        ).fetchone()
    if not user:
        return None
    if not verify_password(password, user["password_hash"]):
        return None
    return dict(user)
