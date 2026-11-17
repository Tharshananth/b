"""Authentication dependencies and logging utilities for FastAPI endpoints."""
import logging
from typing import Optional
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session
import jwt

from database.connection import get_db
from database.models import AdminUser, AdminActionLog
from .security import decode_access_token, AuthConfigError

logger = logging.getLogger(__name__)

bearer_scheme = HTTPBearer(auto_error=False)


def _extract_token(creds: Optional[HTTPAuthorizationCredentials]) -> Optional[str]:
    """
    Extract token from authorization credentials.
    Isolated in its own function so extraction can be switched to httpOnly cookies
    without changing endpoint logic.
    """
    if creds and creds.scheme.lower() == "bearer":
        return creds.credentials
    return None


def client_ip(request: Request) -> Optional[str]:
    """Return client IP address or None."""
    if request and request.client:
        return request.client.host
    return None


def log_admin_action(
    db: Session,
    *,
    username: str,
    action: str,
    target: Optional[str] = None,
    succeeded: bool = True,
    detail: Optional[str] = None,
    ip: Optional[str] = None,
) -> None:
    """
    Best-effort audit logging for admin actions.
    Catches all exceptions and rolls back so failures in logging never break the request.
    """
    try:
        log_entry = AdminActionLog(
            username=username,
            action=action,
            target=target,
            succeeded=succeeded,
            detail=detail,
            client_ip=ip,
        )
        db.add(log_entry)
        db.commit()
    except Exception as e:
        try:
            db.rollback()
        except Exception:
            pass
        logger.error(f"Failed to log admin action ({action} by {username}): {e}", exc_info=True)


def require_admin(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> AdminUser:
    """
    FastAPI dependency enforcing admin authentication.
    Validates token presence, signatures, claims, and live DB status on every request.
    """
    token = _extract_token(creds)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        claims = decode_access_token(token)
    except AuthConfigError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Admin authentication is not configured",
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    username = claims.get("sub")
    user = db.query(AdminUser).filter(AdminUser.username == username).first()
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Account disabled or removed",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if user.role not in {"admin"}:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required",
        )

    return user

