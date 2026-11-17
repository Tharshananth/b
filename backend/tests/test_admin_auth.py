"""
Comprehensive pytest suite for BSAI Chatbot admin-only authentication.
Tests all security requirements without external dependencies (no Chroma, no LLMs, no API keys).
"""
import os
import sys
import json
import base64
import shutil
import tempfile
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock
import pytest

# Ensure backend directory is in sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

# ─────────────────────────────────────────────────────────────────────────────
# 1. Stubs for heavy dependencies before any router imports
# ─────────────────────────────────────────────────────────────────────────────

# Stub vector_db.retriever
mock_retriever_mod = types.ModuleType("vector_db.retriever")
mock_retriever_instance = MagicMock()
mock_vector_store = MagicMock()
mock_vector_store.get_document_count.return_value = 3
mock_vector_store.add_documents.return_value = 5
mock_vector_store.delete_collection.return_value = None
mock_retriever_instance.get_vector_store.return_value = mock_vector_store

mock_retriever_mod.get_retriever = MagicMock(return_value=mock_retriever_instance)
mock_retriever_mod.reset_retriever = MagicMock()
sys.modules["vector_db.retriever"] = mock_retriever_mod

# Stub utils.document_loader
mock_doc_loader_mod = types.ModuleType("utils.document_loader")
mock_doc_loader_cls = MagicMock()
mock_doc = MagicMock()
mock_doc.page_content = "Sample knowledge-base text"
mock_doc.metadata = {"source": "test.txt"}
mock_doc_loader_cls.load_document.return_value = [mock_doc]
mock_doc_loader_cls.load_directory.return_value = [mock_doc]
mock_doc_loader_mod.DocumentLoader = mock_doc_loader_cls
sys.modules["utils.document_loader"] = mock_doc_loader_mod

# Stub llm.factory
mock_llm_factory_mod = types.ModuleType("llm.factory")
mock_factory = MagicMock()
mock_factory.get_provider_info.return_value = [
    {"name": "gemini", "model": "gemini-2.5-flash", "temperature": 0.4, "max_tokens": 1500}
]
mock_factory.get_available_providers.return_value = ["gemini"]
mock_llm_factory_mod.get_llm_factory = MagicMock(return_value=mock_factory)
sys.modules["llm.factory"] = mock_llm_factory_mod

# Register routers as a bare package so routers/__init__.py (which imports chat router) does not execute
routers_dir = str(backend_dir / "routers")
if "routers" not in sys.modules:
    bare_routers = types.ModuleType("routers")
    bare_routers.__path__ = [routers_dir]
    sys.modules["routers"] = bare_routers

# ─────────────────────────────────────────────────────────────────────────────
# 2. Temp directory setup & Environment configuration
# ─────────────────────────────────────────────────────────────────────────────

TEST_JWT_SECRET = "test-secret-at-least-32-chars-long-abcdef123456"
os.environ["JWT_SECRET"] = TEST_JWT_SECRET
os.environ["ADMIN_TOKEN_TTL_MINUTES"] = "60"

# Setup isolated temp working directory for SQLite relative path data/database/feedback.db
temp_work_dir = Path(tempfile.mkdtemp(prefix="bsai_test_"))
orig_cwd = os.getcwd()
os.chdir(temp_work_dir)

# Copy config.yaml into temp working directory
shutil.copy(backend_dir / "config.yaml", temp_work_dir / "config.yaml")

# Create directories expected by config.yaml
(temp_work_dir / "data" / "database").mkdir(parents=True, exist_ok=True)
(temp_work_dir / "data" / "documents").mkdir(parents=True, exist_ok=True)
(temp_work_dir / "uploads").mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Import app dependencies & Initialize Database
# ─────────────────────────────────────────────────────────────────────────────
import jwt
from fastapi import FastAPI
from fastapi.testclient import TestClient
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from database.connection import init_db, SessionLocal, engine
from database.models import AdminUser, AdminActionLog, DataAccessLog
from auth.security import hash_password, create_access_token, decode_access_token
from routers.auth import router as auth_router, limiter as auth_limiter
from routers.config import router as config_router
from routers.documents import router as documents_router

# Initialize database schema in temp directory
init_db()

# Build minimal FastAPI test app
app = FastAPI(title="BSAI Admin Test App")
app.state.limiter = auth_limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.include_router(auth_router)
app.include_router(config_router)
app.include_router(documents_router)

client = TestClient(app)


@pytest.fixture(scope="session", autouse=True)
def teardown_test_environment():
    """Cleanup temp working directory after all tests finish."""
    yield
    try:
        engine.dispose()
    except Exception:
        pass
    os.chdir(orig_cwd)
    try:
        shutil.rmtree(temp_work_dir, ignore_errors=True)
    except Exception:
        pass


@pytest.fixture(autouse=True)
def reset_state_between_tests():
    """Ensure limiter is cleared and default JWT_SECRET is restored between tests."""
    os.environ["JWT_SECRET"] = TEST_JWT_SECRET
    os.environ["ADMIN_TOKEN_TTL_MINUTES"] = "60"
    auth_limiter.reset()
    yield
    auth_limiter.reset()


@pytest.fixture
def db_session():
    """Provide a direct database session for test assertions and fixture setup."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def seed_admin_user(db_session):
    """Seed a standard active admin user."""
    user = db_session.query(AdminUser).filter(AdminUser.username == "admin").first()
    if not user:
        user = AdminUser(
            username="admin",
            password_hash=hash_password("adminpassword123"),
            role="admin",
            is_active=True,
        )
        db_session.add(user)
        db_session.commit()
    else:
        user.password_hash = hash_password("adminpassword123")
        user.role = "admin"
        user.is_active = True
        db_session.commit()
    return user


@pytest.fixture
def admin_token(seed_admin_user):
    """Provide a valid bearer token for the seeded admin user."""
    return create_access_token(seed_admin_user.username, seed_admin_user.role)


# ─────────────────────────────────────────────────────────────────────────────
# Required Tests (1 to 15)
# ─────────────────────────────────────────────────────────────────────────────

def test_01_correct_login_claims_and_expires(seed_admin_user):
    """1. Correct login returns a token whose decoded claims have the right sub and role. expires_in = 3600."""
    response = client.post(
        "/api/auth/login",
        json={"username": "admin", "password": "adminpassword123"},
    )
    assert response.status_code == 200, response.text
    data = response.json()

    assert data["token_type"] == "bearer"
    assert data["expires_in"] == 3600
    assert "access_token" in data

    claims = decode_access_token(data["access_token"])
    assert claims["sub"] == "admin"
    assert claims["role"] == "admin"
    assert claims["iss"] == "bsai-chatbot"
    assert "iat" in claims
    assert "exp" in claims


def test_02_username_case_and_whitespace_insensitive(seed_admin_user):
    """2. The username is case- and whitespace-insensitive."""
    response = client.post(
        "/api/auth/login",
        json={"username": "   ADMIN   ", "password": "adminpassword123"},
    )
    assert response.status_code == 200
    claims = decode_access_token(response.json()["access_token"])
    assert claims["sub"] == "admin"


def test_03_wrong_password_and_unknown_user_identical_401(seed_admin_user):
    """3. A wrong password and an unknown user return identical 401 bodies."""
    # Attempt 1: wrong password for existing user
    resp_wrong_pw = client.post(
        "/api/auth/login",
        json={"username": "admin", "password": "wrong_password_xyz"},
    )
    # Attempt 2: unknown username
    resp_unknown_user = client.post(
        "/api/auth/login",
        json={"username": "definitely_unknown_user", "password": "anypassword"},
    )

    assert resp_wrong_pw.status_code == 401
    assert resp_unknown_user.status_code == 401
    assert resp_wrong_pw.json() == resp_unknown_user.json() == {"detail": "Invalid username or password"}


def test_04_disabled_user_cannot_login(db_session):
    """4. A disabled user cannot log in."""
    user = db_session.query(AdminUser).filter(AdminUser.username == "inactive_admin").first()
    if not user:
        user = AdminUser(
            username="inactive_admin",
            password_hash=hash_password("adminpassword123"),
            role="admin",
            is_active=False,
        )
        db_session.add(user)
    else:
        user.is_active = False
    db_session.commit()

    response = client.post(
        "/api/auth/login",
        json={"username": "inactive_admin", "password": "adminpassword123"},
    )
    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid username or password"}


def test_05_rate_limit_sixth_attempt_returns_429(seed_admin_user):
    """5. The 6th login attempt within a minute returns 429."""
    auth_limiter.reset()

    # First 5 attempts within a minute
    for _ in range(5):
        resp = client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "adminpassword123"},
        )
        assert resp.status_code in (200, 401)

    # 6th attempt must be rate-limited
    sixth_resp = client.post(
        "/api/auth/login",
        json={"username": "admin", "password": "adminpassword123"},
    )
    assert sixth_resp.status_code == 429


def test_06_failed_and_successful_logins_logged_in_audit(db_session, seed_admin_user):
    """6. Failed and successful logins both appear in admin_action_log."""
    initial_dal = db_session.query(DataAccessLog).count()

    # Trigger a successful login
    client.post(
        "/api/auth/login",
        json={"username": "admin", "password": "adminpassword123"},
    )
    # Trigger a failed login
    client.post(
        "/api/auth/login",
        json={"username": "admin", "password": "incorrect_password"},
    )

    success_logs = (
        db_session.query(AdminActionLog)
        .filter(
            AdminActionLog.action == "login",
            AdminActionLog.username == "admin",
            AdminActionLog.succeeded.is_(True),
        )
        .all()
    )
    assert len(success_logs) >= 1

    failure_logs = (
        db_session.query(AdminActionLog)
        .filter(
            AdminActionLog.action == "login",
            AdminActionLog.username == "admin",
            AdminActionLog.succeeded.is_(False),
        )
        .all()
    )
    assert len(failure_logs) >= 1

    # Verify no DataAccessLog rows were created for admin actions
    assert db_session.query(DataAccessLog).count() == initial_dal


def test_07_all_documents_and_config_routes_return_401_without_token():
    """7. Every documents and config route returns 401 without a token (including upload)."""
    # Documents routes
    r_docs_list = client.get("/api/documents/")
    assert r_docs_list.status_code == 401
    assert "Bearer" in r_docs_list.headers.get("WWW-Authenticate", "")

    r_docs_upload = client.post(
        "/api/documents/upload",
        files={"file": ("test.txt", b"dummy content", "text/plain")},
    )
    assert r_docs_upload.status_code == 401

    r_docs_delete = client.delete("/api/documents/test.txt")
    assert r_docs_delete.status_code == 401

    r_docs_refresh = client.post("/api/documents/refresh")
    assert r_docs_refresh.status_code == 401

    # Config routes
    r_cfg_get = client.get("/api/config/")
    assert r_cfg_get.status_code == 401

    r_cfg_providers = client.get("/api/config/providers")
    assert r_cfg_providers.status_code == 401


def test_08_all_documents_and_config_routes_accepted_with_valid_token(admin_token):
    """8. Every documents and config route is accepted with a valid token."""
    headers = {"Authorization": f"Bearer {admin_token}"}

    # 1. GET /api/documents/
    r_list = client.get("/api/documents/", headers=headers)
    assert r_list.status_code == 200
    assert isinstance(r_list.json(), list)

    # 2. POST /api/documents/upload
    r_upload = client.post(
        "/api/documents/upload",
        headers=headers,
        files={"file": ("sample.txt", b"Sample upload content", "text/plain")},
    )
    assert r_upload.status_code == 200
    assert r_upload.json()["success"] is True

    # 3. DELETE /api/documents/{filename}
    r_delete = client.delete("/api/documents/sample.txt", headers=headers)
    assert r_delete.status_code == 200
    assert r_delete.json()["success"] is True

    # 4. POST /api/documents/refresh
    r_refresh = client.post("/api/documents/refresh", headers=headers)
    assert r_refresh.status_code == 200
    assert r_refresh.json()["success"] is True

    # 5. GET /api/config/
    r_config = client.get("/api/config/", headers=headers)
    assert r_config.status_code == 200
    assert "app" in r_config.json()

    # 6. GET /api/config/providers
    r_prov = client.get("/api/config/providers", headers=headers)
    assert r_prov.status_code == 200
    assert isinstance(r_prov.json(), list)

    # 7. GET /api/auth/me
    r_me = client.get("/api/auth/me", headers=headers)
    assert r_me.status_code == 200
    assert r_me.json() == {"username": "admin", "role": "admin"}


def test_09_garbage_different_secret_alg_none_and_expired_tokens_return_401():
    """9. A garbage token, a token signed with a different secret, an alg: none token and an expired token all return 401."""
    # 1. Garbage token
    r_garbage = client.get(
        "/api/documents/",
        headers={"Authorization": "Bearer not-a-valid-token-string"},
    )
    assert r_garbage.status_code == 401
    assert r_garbage.json() == {"detail": "Invalid token"}

    # 2. Token signed with different secret
    now = datetime.now(timezone.utc)
    diff_payload = {
        "sub": "admin",
        "role": "admin",
        "iss": "bsai-chatbot",
        "iat": now,
        "exp": now + timedelta(hours=1),
    }
    diff_token = jwt.encode(diff_payload, "different-secret-32-chars-long-12345", algorithm="HS256")
    r_diff = client.get(
        "/api/documents/",
        headers={"Authorization": f"Bearer {diff_token}"},
    )
    assert r_diff.status_code == 401
    assert r_diff.json() == {"detail": "Invalid token"}

    # 3. alg: none token
    header_b64 = base64.urlsafe_b64encode(json.dumps({"alg": "none", "typ": "JWT"}).encode()).decode().rstrip("=")
    payload_b64 = base64.urlsafe_b64encode(json.dumps({
        "sub": "admin",
        "role": "admin",
        "iss": "bsai-chatbot",
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(hours=1)).timestamp()),
    }).encode()).decode().rstrip("=")
    alg_none_token = f"{header_b64}.{payload_b64}."

    r_none = client.get(
        "/api/documents/",
        headers={"Authorization": f"Bearer {alg_none_token}"},
    )
    assert r_none.status_code == 401
    assert r_none.json() == {"detail": "Invalid token"}

    # 4. Expired token
    expired_payload = {
        "sub": "admin",
        "role": "admin",
        "iss": "bsai-chatbot",
        "iat": now - timedelta(hours=2),
        "exp": now - timedelta(hours=1),
    }
    expired_token = jwt.encode(expired_payload, TEST_JWT_SECRET, algorithm="HS256")
    r_expired = client.get(
        "/api/documents/",
        headers={"Authorization": f"Bearer {expired_token}"},
    )
    assert r_expired.status_code == 401
    assert r_expired.json() == {"detail": "Token expired"}


def test_10_disabling_user_makes_existing_token_fail_immediately(db_session):
    """10. Disabling a user makes their existing token return 401 immediately."""
    user = db_session.query(AdminUser).filter(AdminUser.username == "temporary_admin").first()
    if not user:
        user = AdminUser(
            username="temporary_admin",
            password_hash=hash_password("adminpassword123"),
            role="admin",
            is_active=True,
        )
        db_session.add(user)
    else:
        user.is_active = True
    db_session.commit()

    token = create_access_token(user.username, user.role)
    headers = {"Authorization": f"Bearer {token}"}

    # Verify token works when active
    r1 = client.get("/api/documents/", headers=headers)
    assert r1.status_code == 200

    # Disable user in DB
    user.is_active = False
    db_session.commit()

    # Existing token must immediately be rejected
    r2 = client.get("/api/documents/", headers=headers)
    assert r2.status_code == 401
    assert r2.json() == {"detail": "Account disabled or removed"}


def test_11_user_with_role_viewer_gets_403(db_session):
    """11. A user with role viewer gets 403."""
    user = db_session.query(AdminUser).filter(AdminUser.username == "viewer_user").first()
    if not user:
        user = AdminUser(
            username="viewer_user",
            password_hash=hash_password("adminpassword123"),
            role="viewer",
            is_active=True,
        )
        db_session.add(user)
    else:
        user.role = "viewer"
        user.is_active = True
    db_session.commit()

    viewer_token = create_access_token(user.username, user.role)
    headers = {"Authorization": f"Bearer {viewer_token}"}

    response = client.get("/api/documents/", headers=headers)
    assert response.status_code == 403
    assert response.json() == {"detail": "Admin role required"}


def test_12_empty_jwt_secret_makes_protected_routes_return_503(admin_token):
    """12. An empty JWT_SECRET makes protected routes return 503."""
    headers = {"Authorization": f"Bearer {admin_token}"}

    # Set secret to empty string
    os.environ["JWT_SECRET"] = ""
    r_empty = client.get("/api/documents/", headers=headers)
    assert r_empty.status_code == 503
    assert r_empty.json() == {"detail": "Admin authentication is not configured"}

    # Set secret to less than 32 chars
    os.environ["JWT_SECRET"] = "short_secret_under_32_chars"
    r_short = client.get("/api/documents/", headers=headers)
    assert r_short.status_code == 503
    assert r_short.json() == {"detail": "Admin authentication is not configured"}

    # Restore valid secret
    os.environ["JWT_SECRET"] = TEST_JWT_SECRET


def test_13_removed_endpoints_return_404_or_405(admin_token):
    """13. POST /api/config/provider/openai and GET /api/config/system-prompt return 404 (or 405)."""
    headers = {"Authorization": f"Bearer {admin_token}"}

    # POST /api/config/provider/openai
    r_post_prov = client.post("/api/config/provider/openai", headers=headers)
    assert r_post_prov.status_code in (404, 405)

    # GET /api/config/system-prompt
    r_get_prompt = client.get("/api/config/system-prompt", headers=headers)
    assert r_get_prompt.status_code in (404, 405)


def test_14_uploading_evil_exe_returns_400(admin_token):
    """14. Uploading evil.exe returns 400."""
    headers = {"Authorization": f"Bearer {admin_token}"}
    response = client.post(
        "/api/documents/upload",
        headers=headers,
        files={"file": ("evil.exe", b"malicious binary payload", "application/octet-stream")},
    )
    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid filename or unsupported file type"}


def test_15_successful_upload_logged_in_audit_with_username(db_session, admin_token):
    """15. A successful upload is logged in admin_action_log with the admin's username."""
    initial_dal = db_session.query(DataAccessLog).count()
    headers = {"Authorization": f"Bearer {admin_token}"}
    target_filename = "audit_upload_test.txt"

    response = client.post(
        "/api/documents/upload",
        headers=headers,
        files={"file": (target_filename, b"Audit log verification content", "text/plain")},
    )
    assert response.status_code == 200

    log_entry = (
        db_session.query(AdminActionLog)
        .filter(
            AdminActionLog.action == "upload",
            AdminActionLog.target == target_filename,
            AdminActionLog.username == "admin",
            AdminActionLog.succeeded.is_(True),
        )
        .first()
    )
    assert log_entry is not None
    assert log_entry.username == "admin"
    assert log_entry.target == target_filename
    assert log_entry.succeeded is True

    # Verify no DataAccessLog rows were created for admin upload events
    assert db_session.query(DataAccessLog).count() == initial_dal
