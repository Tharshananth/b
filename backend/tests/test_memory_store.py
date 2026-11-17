"""
Pytest suite for conversation memory in Redis and in-process stores.
Tests codec, trimming, sliding TTL, multi-worker sharing, fail-open, circuit-breaking,
and endpoint authentication protection.
"""
import base64
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time
import types
import uuid
from dataclasses import fields
from pathlib import Path
import pytest
import redis

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

# ─────────────────────────────────────────────────────────────────────────────
# 1. Module isolation: load llm.base without running llm/__init__.py
# ─────────────────────────────────────────────────────────────────────────────
llm_dir = str(backend_dir / "llm")
if "llm" not in sys.modules:
    bare_llm = types.ModuleType("llm")
    bare_llm.__path__ = [llm_dir]
    sys.modules["llm"] = bare_llm
else:
    bare_llm = sys.modules["llm"]

if "llm.base" in sys.modules:
    llm_base_mod = sys.modules["llm.base"]
else:
    spec = importlib.util.spec_from_file_location("llm.base", backend_dir / "llm" / "base.py")
    llm_base_mod = importlib.util.module_from_spec(spec)
    sys.modules["llm.base"] = llm_base_mod
    bare_llm.base = llm_base_mod
    spec.loader.exec_module(llm_base_mod)

from llm.base import Message, ToolCall

# Stub llm.factory so agent/orchestrator imports do not initialize LLMs
mock_factory_mod = types.ModuleType("llm.factory")
mock_factory_mod.LLMFactory = object
mock_factory_mod.get_llm_factory = lambda: types.SimpleNamespace(
    get_available_providers=lambda: ["gemini"],
    get_provider_info=lambda: [],
)
sys.modules["llm.factory"] = mock_factory_mod

# Stub vector_db.retriever
if "vector_db.retriever" not in sys.modules:
    mock_retriever_mod = types.ModuleType("vector_db.retriever")
    mock_retriever_mod.get_retriever = lambda: None
    mock_retriever_mod.reset_retriever = lambda: None
    sys.modules["vector_db.retriever"] = mock_retriever_mod

# Register bare routers package
routers_dir = str(backend_dir / "routers")
if "routers" not in sys.modules:
    bare_routers = types.ModuleType("routers")
    bare_routers.__path__ = [routers_dir]
    sys.modules["routers"] = bare_routers
else:
    bare_routers = sys.modules["routers"]

# ─────────────────────────────────────────────────────────────────────────────
# 2. Temp working directory setup for relative paths and config.yaml
# ─────────────────────────────────────────────────────────────────────────────
temp_work_dir = Path(tempfile.mkdtemp(prefix="bsai_mem_test_"))
orig_cwd = os.getcwd()
os.chdir(temp_work_dir)
shutil.copy(backend_dir / "config.yaml", temp_work_dir / "config.yaml")
(temp_work_dir / "data" / "database").mkdir(parents=True, exist_ok=True)

# Set test JWT secret
TEST_JWT_SECRET = "test-secret-at-least-32-chars-long-abcdef123456"
os.environ["JWT_SECRET"] = TEST_JWT_SECRET

from database.connection import init_db, SessionLocal, engine
init_db()

from auth.security import create_access_token, hash_password
from database.models import AdminUser

from memory import codec
from memory.store import (
    MemoryStore,
    InProcessMemoryStore,
    RedisMemoryStore,
    get_memory_store,
    _reset_memory_store,
)

from routers.chat import router as chat_router
import routers.chat as _chat_module
bare_routers.chat = _chat_module
from fastapi import FastAPI
from fastapi.testclient import TestClient

app = FastAPI(title="Memory Test App")
app.include_router(chat_router)
test_client = TestClient(app)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Helpers & Fixtures
# ─────────────────────────────────────────────────────────────────────────────
TEST_PREFIX = f"bsai:test:{uuid.uuid4().hex[:8]}:"
DEFAULT_REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


def check_redis(url: str) -> bool:
    try:
        r = redis.Redis.from_url(url, socket_connect_timeout=1, socket_timeout=1)
        return bool(r.ping())
    except Exception:
        return False


HAS_REDIS = check_redis(DEFAULT_REDIS_URL)


@pytest.fixture(scope="session", autouse=True)
def cleanup_resources():
    yield
    # Cleanup Redis test keys
    if HAS_REDIS:
        try:
            r = redis.Redis.from_url(DEFAULT_REDIS_URL, socket_connect_timeout=2, socket_timeout=2)
            cursor = 0
            while True:
                cursor, keys = r.scan(cursor=cursor, match=f"{TEST_PREFIX}*", count=500)
                if keys:
                    r.delete(*keys)
                if cursor == 0:
                    break
        except Exception:
            pass
    # Cleanup SQLite engine & temp directory
    try:
        engine.dispose()
    except Exception:
        pass
    os.chdir(orig_cwd)
    try:
        shutil.rmtree(temp_work_dir, ignore_errors=True)
    except Exception:
        pass


def make_tool_exchange(turn: int, session: str = "test") -> list:
    call_id = f"call_{session}_{turn}"
    return [
        Message(role="user", content=f"[{session}] turn {turn}: status of patient 2013?"),
        Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(
                    id=call_id,
                    name="get_job_status",
                    arguments={"patient_id": "2013", "turn": turn},
                    provider_data={"thought_signature": b"\x00\xffsig"},
                )
            ],
        ),
        Message(
            role="tool",
            content='{"status": "completed"}',
            tool_call_id=call_id,
            name="get_job_status",
        ),
        Message(role="assistant", content=f"[{session}] turn {turn}: completed."),
    ]


def make_plain_turn(turn: int, session: str = "test") -> list:
    return [
        Message(role="user", content=f"[{session}] user message {turn}"),
        Message(role="assistant", content=f"[{session}] assistant reply {turn}"),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# In-process tests (1 - 4)
# ─────────────────────────────────────────────────────────────────────────────

def test_01_codec_round_trip():
    """1. Codec round trip: messages are equal, and thought_signature is still bytes."""
    convo = make_tool_exchange(1) + make_plain_turn(2)
    raw = codec.dumps(convo)
    loaded = codec.loads(raw)

    assert loaded == convo
    sig = loaded[1].tool_calls[0].provider_data["thought_signature"]
    assert isinstance(sig, bytes)
    assert sig == b"\x00\xffsig"


def test_02_plain_json_dumps_raises_type_error():
    """2. Plain json.dumps of a message with bytes raises TypeError (documents why the codec exists)."""
    msg = Message(
        role="assistant",
        tool_calls=[
            ToolCall(
                id="call_1",
                name="test_tool",
                arguments={},
                provider_data={"thought_signature": b"\x00\xffsig"},
            )
        ],
    )
    with pytest.raises(TypeError):
        json.dumps(msg.__dict__)


def test_03_trim_keeps_last_k_user_turns_with_tool_turns():
    """3. trim: 15 turns with k=10 keeps 10 user turns, starts with a user message, and all tool call ids match results."""
    messages = []
    for t in range(1, 16):
        messages += make_tool_exchange(t) if t % 2 == 1 else make_plain_turn(t)

    trimmed = codec.trim(messages, k=10)
    users = [m for m in trimmed if m.role == "user"]
    assert len(users) == 10
    assert trimmed[0].role == "user"

    call_ids = {tc.id for m in trimmed if m.tool_calls for tc in m.tool_calls}
    tool_result_ids = {m.tool_call_id for m in trimmed if m.role == "tool"}
    assert call_ids == tool_result_ids


def test_04_in_process_memory_store():
    """4. InProcessMemoryStore: save then load round trip, save trims to k, clear returns True then False, exists."""
    store = InProcessMemoryStore(k=3)
    session_id = "test_in_proc_session"

    convo = make_plain_turn(1) + make_plain_turn(2)
    store.save(session_id, convo)
    assert store.exists(session_id) is True
    assert store.load(session_id) == convo

    # Save trims to k=3 user turns
    long_convo = []
    for i in range(1, 6):
        long_convo += make_plain_turn(i)
    store.save(session_id, long_convo)
    loaded = store.load(session_id)
    users = [m for m in loaded if m.role == "user"]
    assert len(users) == 3
    assert users[0].content.endswith("3")

    # clear returns True then False
    assert store.clear(session_id) is True
    assert store.clear(session_id) is False
    assert store.exists(session_id) is False
    assert store.load(session_id) == []


# ─────────────────────────────────────────────────────────────────────────────
# Real Redis tests (5 - 9)
# ─────────────────────────────────────────────────────────────────────────────

def test_05_redis_save_load_round_trip():
    """5. Save/load round trip through Redis, with bytes preserved."""
    if not HAS_REDIS:
        pytest.skip(f"Redis not available at {DEFAULT_REDIS_URL}")

    store = RedisMemoryStore(
        url=DEFAULT_REDIS_URL,
        k=10,
        ttl_seconds=60,
        key_prefix=TEST_PREFIX,
    )
    session_id = f"sess_{uuid.uuid4().hex[:6]}"
    convo = make_tool_exchange(1, session=session_id) + make_plain_turn(2, session=session_id)

    store.save(session_id, convo)
    loaded = store.load(session_id)
    assert loaded == convo
    sig = loaded[1].tool_calls[0].provider_data["thought_signature"]
    assert isinstance(sig, bytes)
    assert sig == b"\x00\xffsig"


def test_06_redis_ttl_and_sliding_expiry():
    """6. After save, the TTL is between 1 and configured. A second save resets it: ttl=3s, save, sleep 2, save, sleep 2 -> exists; sleep 2 more -> gone."""
    if not HAS_REDIS:
        pytest.skip(f"Redis not available at {DEFAULT_REDIS_URL}")

    store = RedisMemoryStore(
        url=DEFAULT_REDIS_URL,
        k=10,
        ttl_seconds=3,
        key_prefix=TEST_PREFIX,
    )
    session_id = f"sess_ttl_{uuid.uuid4().hex[:6]}"
    r = redis.Redis.from_url(DEFAULT_REDIS_URL)

    store.save(session_id, make_plain_turn(1))
    key = f"{TEST_PREFIX}{session_id}"
    ttl = r.ttl(key)
    assert 1 <= ttl <= 3

    time.sleep(2)
    # Second save resets the clock (sliding expiry)
    store.save(session_id, make_plain_turn(1) + make_plain_turn(2))
    time.sleep(2)
    assert store.exists(session_id) is True

    # Idle for 2 more seconds -> expires
    time.sleep(2)
    assert store.exists(session_id) is False
    assert store.load(session_id) == []


def test_07_two_separate_redis_stores_see_same_session():
    """7. Two separate RedisMemoryStore instances (simulating two workers) see the same session."""
    if not HAS_REDIS:
        pytest.skip(f"Redis not available at {DEFAULT_REDIS_URL}")

    worker_a = RedisMemoryStore(url=DEFAULT_REDIS_URL, k=10, ttl_seconds=60, key_prefix=TEST_PREFIX)
    worker_b = RedisMemoryStore(url=DEFAULT_REDIS_URL, k=10, ttl_seconds=60, key_prefix=TEST_PREFIX)
    session_id = f"sess_workers_{uuid.uuid4().hex[:6]}"

    convo = make_plain_turn(1)
    worker_a.save(session_id, convo)

    assert worker_b.exists(session_id) is True
    loaded_b = worker_b.load(session_id)
    assert loaded_b == convo

    worker_b.clear(session_id)
    assert worker_a.exists(session_id) is False


def test_08_redis_save_trims_to_k():
    """8. Save trims to k in Redis."""
    if not HAS_REDIS:
        pytest.skip(f"Redis not available at {DEFAULT_REDIS_URL}")

    store = RedisMemoryStore(url=DEFAULT_REDIS_URL, k=3, ttl_seconds=60, key_prefix=TEST_PREFIX)
    session_id = f"sess_trim_{uuid.uuid4().hex[:6]}"

    long_convo = []
    for i in range(1, 6):
        long_convo += make_plain_turn(i)

    store.save(session_id, long_convo)
    loaded = store.load(session_id)
    users = [m for m in loaded if m.role == "user"]
    assert len(users) == 3
    assert users[0].content.endswith("3")


def test_09_redis_corrupted_value_handled():
    """9. A corrupted value (manually SET garbage) -> load returns [] and the key is deleted."""
    if not HAS_REDIS:
        pytest.skip(f"Redis not available at {DEFAULT_REDIS_URL}")

    store = RedisMemoryStore(url=DEFAULT_REDIS_URL, k=10, ttl_seconds=60, key_prefix=TEST_PREFIX)
    session_id = f"sess_corrupt_{uuid.uuid4().hex[:6]}"
    key = f"{TEST_PREFIX}{session_id}"

    r = redis.Redis.from_url(DEFAULT_REDIS_URL)
    r.set(key, "not a valid json value", ex=60)

    loaded = store.load(session_id)
    assert loaded == []
    assert r.exists(key) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Failure tests (10 - 14)
# ─────────────────────────────────────────────────────────────────────────────

def test_10_unreachable_redis_load_empty_and_save_no_raise():
    """10. load returns [] and save doesn't raise when Redis is down."""
    store = RedisMemoryStore(
        url="redis://127.0.0.1:1/0",
        k=10,
        ttl_seconds=60,
        connect_timeout_s=0.5,
        op_timeout_s=0.5,
        circuit_breaker_s=30,
    )
    assert store.load("unreachable_session") == []
    store.save("unreachable_session", make_plain_turn(1))
    assert store.clear("unreachable_session") is False
    assert store.exists("unreachable_session") is False


def test_11_circuit_breaker_first_call_and_fast_skips():
    """11. The first call fails within ~2 s. While the breaker is open, the next 20 calls complete in under 50 ms total."""
    store = RedisMemoryStore(
        url="redis://127.0.0.1:1/0",
        k=10,
        ttl_seconds=60,
        connect_timeout_s=0.5,
        op_timeout_s=0.5,
        circuit_breaker_s=30,
    )

    t0 = time.perf_counter()
    res = store.load("first_call")
    first_duration = time.perf_counter() - t0

    assert res == []
    assert first_duration < 3.0

    # Next 20 calls while breaker is open must complete in under 50 ms total
    t1 = time.perf_counter()
    for _ in range(20):
        assert store.load("first_call") == []
    twenty_calls_duration = time.perf_counter() - t1

    assert twenty_calls_duration < 0.05, f"Took {twenty_calls_duration * 1000:.2f} ms"


def test_12_circuit_breaker_cooldown_and_retry():
    """12. After the cooldown (use circuit_breaker_s=1 and sleep 1.1), the store tries Redis again."""
    store = RedisMemoryStore(
        url="redis://127.0.0.1:1/0",
        k=10,
        ttl_seconds=60,
        connect_timeout_s=0.5,
        op_timeout_s=0.5,
        circuit_breaker_s=1,
    )

    store.load("cooldown_test")
    assert store._is_circuit_open() is True

    time.sleep(1.1)
    assert store._is_circuit_open() is False

    t0 = time.perf_counter()
    store.load("cooldown_test")
    duration = time.perf_counter() - t0
    # Proves it attempted to connect to Redis instead of fast-skipping
    assert duration > 0.01


def test_13_status_reports_disconnected_and_circuit_open():
    """13. status() reports connected: False and circuit_open: True after a failure."""
    store = RedisMemoryStore(
        url="redis://127.0.0.1:1/0",
        k=10,
        ttl_seconds=60,
        connect_timeout_s=0.5,
        op_timeout_s=0.5,
        circuit_breaker_s=30,
    )
    store.load("status_test")
    stat = store.status()
    assert stat["backend"] == "redis"
    assert stat["connected"] is False
    assert stat["circuit_open"] is True


def test_14_get_memory_store_fallback_when_no_redis_url():
    """14. get_memory_store() with backend='redis' and no REDIS_URL falls back to in-process and does not raise."""
    old_redis_url = os.environ.pop("REDIS_URL", None)
    try:
        _reset_memory_store()
        store = get_memory_store()
        assert isinstance(store, InProcessMemoryStore)
        assert store.status()["backend"] == "in_process"
    finally:
        if old_redis_url is not None:
            os.environ["REDIS_URL"] = old_redis_url
        _reset_memory_store()


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint tests (15)
# ─────────────────────────────────────────────────────────────────────────────

def test_15_chat_memory_endpoints_require_admin():
    """15. GET /memory/info/x, GET /history/x, DELETE /history/x and POST /clear/x return 401 without a token."""
    session_id = "test_sess_auth_401"

    # All 4 memory management endpoints return 401 without Bearer token
    assert test_client.get(f"/api/chat/memory/info/{session_id}").status_code == 401
    assert test_client.get(f"/api/chat/history/{session_id}").status_code == 401
    assert test_client.delete(f"/api/chat/history/{session_id}").status_code == 401
    assert test_client.post(f"/api/chat/clear/{session_id}").status_code == 401

    # Verify that with valid admin credentials, they are accepted
    db = SessionLocal()
    try:
        admin_user = db.query(AdminUser).filter(AdminUser.username == "admin_memory").first()
        if not admin_user:
            admin_user = AdminUser(
                username="admin_memory",
                password_hash=hash_password("adminpassword123"),
                role="admin",
                is_active=True,
            )
            db.add(admin_user)
            db.commit()
    finally:
        db.close()

    token = create_access_token("admin_memory", "admin")
    headers = {"Authorization": f"Bearer {token}"}

    r_info = test_client.get(f"/api/chat/memory/info/{session_id}", headers=headers)
    assert r_info.status_code == 200
    assert "backend" in r_info.json()

    r_clear = test_client.post(f"/api/chat/clear/{session_id}", headers=headers)
    assert r_clear.status_code == 200

    r_hist = test_client.get(f"/api/chat/history/{session_id}", headers=headers)
    assert r_hist.status_code == 200

    r_del = test_client.delete(f"/api/chat/history/{session_id}", headers=headers)
    assert r_del.status_code == 200

    # Ensure main POST /api/chat/ does NOT require admin auth
    r_chat = test_client.post("/api/chat/", json={})
    assert r_chat.status_code != 401

