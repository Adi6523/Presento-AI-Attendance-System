from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional
import bcrypt
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from ..config import settings
from ..exceptions import BadRequestException, ConflictException, UnauthorizedException
from ..models import User

logger = logging.getLogger("presento.auth_service")


def hash_password(password: str) -> str:
    """Hashes a password securely using bcrypt."""
    pwd_bytes = password.encode("utf-8")[:72]
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(pwd_bytes, salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plaintext password against a stored bcrypt hash."""
    try:
        pwd_bytes = plain_password.encode("utf-8")[:72]
        return bcrypt.checkpw(pwd_bytes, hashed_password.encode("utf-8"))
    except Exception as exc:
        logger.warning("Password verification error: %s", exc)
        return False


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Generates a signed JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.access_token_expire_minutes)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)
    return encoded_jwt


def decode_access_token(token: str) -> dict:
    """Decodes and validates a JWT token."""
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        return payload
    except JWTError as exc:
        raise UnauthorizedException("Invalid or expired authentication token", error_code="INVALID_TOKEN") from exc


def get_user_by_username(db: Session, username: str) -> Optional[User]:
    return db.query(User).filter(User.username == username).first()


def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
    return db.query(User).filter(User.id == user_id).first()


def authenticate_user(db: Session, username: str, password: str) -> User:
    """Authenticates a user by username and password."""
    user = get_user_by_username(db, username)
    if not user:
        raise UnauthorizedException("Invalid username or password", error_code="INVALID_CREDENTIALS")
    if not user.is_active:
        raise UnauthorizedException("User account is inactive", error_code="USER_INACTIVE")
    if not verify_password(password, user.password_hash):
        raise UnauthorizedException("Invalid username or password", error_code="INVALID_CREDENTIALS")
    return user


def create_user(db: Session, username: str, password: str, role: str = "TEACHER") -> User:
    """Creates a new user record."""
    if get_user_by_username(db, username):
        raise ConflictException(f"Username '{username}' already exists", error_code="USERNAME_EXISTS")
    
    if role not in ("ADMIN", "TEACHER"):
        raise BadRequestException(f"Invalid role '{role}'. Must be ADMIN or TEACHER", error_code="INVALID_ROLE")

    user = User(
        username=username,
        password_hash=hash_password(password),
        role=role,
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user
