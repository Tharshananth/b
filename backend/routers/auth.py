"""Authentication router for admin login and profile retrieval."""
import os
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy.orm import Session

from database.connection import get_db
from database.models import AdminUser
from auth.security import (
    verify_password,
    burn_password_check,
    create_access_token,
    AuthConfigError,
)
from auth.dependencies import require_admin, client_ip, log_admin_action

router = APIRouter(prefix="/api/auth", tags=["Auth"])
limiter = Limiter(key_func=get_remote_address)


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=254)
    password: str = Field(..., min_length=1, max_length=128)


@router.post("/login")
@limiter.limit("5/minute")
def login(request: Request, body: LoginRequest, db: Session = Depends(get_db)):
    """
    Authenticate an admin user and return a JWT access token.
    Uses sync def so bcrypt hashing runs without blocking the async event loop.
    Rate limited to 5 attempts per minute per IP.
    """
    normalized_username = body.username.strip().lower()
    ip = client_ip(request)

    user = db.query(AdminUser).filter(AdminUser.username == normalized_username).first()

    if user is None:
        burn_password_check(body.password)
        log_admin_action(
            db,
            username=normalized_username,
            action="login",
            succeeded=False,
            detail="User not found",
            ip=ip,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    if not user.is_active:
        burn_password_check(body.password)
        log_admin_action(
            db,
            username=normalized_username,
            action="login",
            succeeded=False,
            detail="User disabled",
            ip=ip,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    if not verify_password(body.password, user.password_hash):
        log_admin_action(
            db,
            username=normalized_username,
            action="login",
            succeeded=False,
            detail="Invalid password",
            ip=ip,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    try:
        token = create_access_token(user.username, user.role)
    except AuthConfigError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Admin authentication is not configured",
        )

    user.last_login = datetime.utcnow()
    db.commit()

    log_admin_action(
        db,
        username=user.username,
        action="login",
        succeeded=True,
        ip=ip,
    )

    try:
        ttl_minutes = int(os.getenv("ADMIN_TOKEN_TTL_MINUTES", "60"))
    except ValueError:
        ttl_minutes = 60
    expires_in = ttl_minutes * 60

    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": expires_in,
    }


@router.get("/me")
def get_me(admin: AdminUser = Depends(require_admin)):
    """Return currently authenticated admin user details."""
    return {
        "username": admin.username,
        "role": admin.role,
    }

