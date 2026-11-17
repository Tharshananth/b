"""Authentication security utilities: password hashing, verification, and JWT management."""
import os
from datetime import datetime, timedelta, timezone
import bcrypt
import jwt


class AuthConfigError(Exception):
    """Raised when JWT_SECRET is missing or shorter than 32 characters."""
    pass


# Module-level dummy hash for timing burn attacks (valid 12-round bcrypt hash)
_DUMMY_HASH = "$2b$12$Xym15b28TBbyj8fw57FUJeCm9YHOcqkU135hX.6/0n0komNUNX7Em"


def hash_password(password: str) -> str:
    """
    Hash a password using bcrypt with 12 rounds.
    Raises ValueError if the password exceeds 72 bytes.
    """
    pwd_bytes = password.encode("utf-8")
    if len(pwd_bytes) > 72:
        raise ValueError("Password cannot exceed 72 bytes")
    return bcrypt.hashpw(pwd_bytes, bcrypt.gensalt(rounds=12)).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """
    Verify a password against a bcrypt hash.
    Returns False on any ValueError.
    """
    try:
        pwd_bytes = password.encode("utf-8")
        if len(pwd_bytes) > 72:
            return False
        hash_bytes = password_hash.encode("utf-8") if isinstance(password_hash, str) else password_hash
        return bcrypt.checkpw(pwd_bytes, hash_bytes)
    except ValueError:
        return False


def burn_password_check(password: str) -> None:
    """
    Verify password against a module-level dummy hash to ensure constant response
    timing when a user does not exist or is inactive.
    """
    verify_password(password, _DUMMY_HASH)


def _get_jwt_secret() -> str:
    """Retrieve and validate the JWT secret from environment."""
    secret = os.getenv("JWT_SECRET")
    if not secret or len(secret) < 32:
        raise AuthConfigError("Admin authentication is not configured")
    return secret


def create_access_token(username: str, role: str) -> str:
    """
    Create a signed HS256 JWT access token with standard claims.
    Uses timezone-aware UTC datetimes.
    """
    secret = _get_jwt_secret()
    try:
        ttl_minutes = int(os.getenv("ADMIN_TOKEN_TTL_MINUTES", "60"))
    except ValueError:
        ttl_minutes = 60

    now = datetime.now(timezone.utc)
    exp = now + timedelta(minutes=ttl_minutes)

    payload = {
        "sub": username,
        "role": role,
        "iss": "bsai-chatbot",
        "iat": now,
        "exp": exp,
    }

    return jwt.encode(payload, secret, algorithm="HS256")


def decode_access_token(token: str) -> dict:
    """
    Decode and validate an HS256 JWT access token.
    Enforces HS256 algorithm, issuer, and required claims.
    """
    secret = _get_jwt_secret()
    return jwt.decode(
        token,
        secret,
        algorithms=["HS256"],
        issuer="bsai-chatbot",
        options={"require": ["sub", "role", "exp", "iat", "iss"]},
    )

