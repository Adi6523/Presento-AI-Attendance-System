from __future__ import annotations

from typing import Callable, List, Optional
from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from ..config import settings
from ..database import get_db
from ..exceptions import ForbiddenException, UnauthorizedException
from ..models import User
from ..services.auth_service import decode_access_token, get_user_by_id

security = HTTPBearer(auto_error=False)
ai_security = HTTPBearer(auto_error=False)


def verify_ai_api_key(
    credentials: HTTPAuthorizationCredentials = Depends(ai_security),
) -> None:
    """
    Bearer token authentication for the AI Vision engine.
    Validates against PRESENTO_AI_API_KEY (separate from frontend JWT).
    """
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "success": False,
                "error": "AI_KEY_MISSING",
                "message": "AI service Authorization header with Bearer token is required",
            },
        )
    if credentials.credentials != settings.presento_ai_api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "success": False,
                "error": "AI_KEY_INVALID",
                "message": "Invalid AI service API key",
            },
        )



def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
) -> User:
    """Extracts and validates JWT bearer token from request header."""
    if not credentials or not credentials.credentials:
        raise UnauthorizedException("Authorization header with Bearer token is required", error_code="TOKEN_MISSING")

    token = credentials.credentials
    payload = decode_access_token(token)
    user_id = payload.get("sub")
    if not user_id:
        raise UnauthorizedException("Invalid token payload", error_code="INVALID_PAYLOAD")

    user = get_user_by_id(db, int(user_id))
    if not user:
        raise UnauthorizedException("User no longer exists", error_code="USER_NOT_FOUND")

    if not user.is_active:
        raise UnauthorizedException("User account is inactive", error_code="USER_INACTIVE")

    return user


def get_current_user_or_ai_key(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
) -> Optional[User]:
    """
    Dual-authentication dependency:
    Accepts either a valid teacher/admin JWT bearer token,
    OR a valid PRESENTO_AI_API_KEY bearer token (for AI session discovery).
    """
    if not credentials or not credentials.credentials:
        raise UnauthorizedException("Authorization header with Bearer token is required", error_code="TOKEN_MISSING")

    token = credentials.credentials
    # Allow AI Engine via PRESENTO_AI_API_KEY
    if token == settings.presento_ai_api_key:
        return None

    # Validate Teacher JWT
    payload = decode_access_token(token)
    user_id = payload.get("sub")
    if not user_id:
        raise UnauthorizedException("Invalid token payload", error_code="INVALID_PAYLOAD")

    user = get_user_by_id(db, int(user_id))
    if not user:
        raise UnauthorizedException("User no longer exists", error_code="USER_NOT_FOUND")

    if not user.is_active:
        raise UnauthorizedException("User account is inactive", error_code="USER_INACTIVE")

    return user


def require_role(*roles: str) -> Callable:
    """Dependency factory restricting access to specified roles (e.g., ADMIN, TEACHER)."""
    def role_checker(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in roles:
            raise ForbiddenException(
                f"Access denied for role '{current_user.role}'. Required: {roles}",
                error_code="FORBIDDEN_ROLE",
            )
        return current_user

    return role_checker
