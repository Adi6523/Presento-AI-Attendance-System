from __future__ import annotations

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import User
from ..schemas.auth import LoginRequest, TokenResponse, UserBrief
from ..services.auth_service import authenticate_user, create_access_token
from .dependencies import get_current_user

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


@router.post(
    "/login",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
    summary="User Login",
    description="Authenticates user with username and password, returning a JWT access token.",
)
def login(request: LoginRequest, db: Session = Depends(get_db)):
    user = authenticate_user(db, request.username, request.password)
    access_token = create_access_token(
        data={"sub": str(user.id), "username": user.username, "role": user.role}
    )
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": user,
    }


@router.get(
    "/me",
    response_model=UserBrief,
    status_code=status.HTTP_200_OK,
    summary="Current User Profile",
    description="Returns the profile of the currently authenticated user. Never returns password hashes.",
)
def get_me(current_user: User = Depends(get_current_user)):
    return current_user


@router.post(
    "/logout",
    status_code=status.HTTP_200_OK,
    summary="User Logout",
    description="Stateless logout invalidation endpoint for frontend session clearing.",
)
def logout(current_user: User = Depends(get_current_user)):
    return {
        "success": True,
        "message": f"User '{current_user.username}' successfully logged out",
    }
