"""
Conversation memory storage implementations: InProcessMemoryStore and RedisMemoryStore.
Provides pluggable, process-safe, fail-open conversation history with TTL and circuit-breaking.
"""
from abc import ABC, abstractmethod
import logging
import os
import threading
import time
from typing import Dict, List, Optional
import redis

from llm.base import Message
from . import codec

logger = logging.getLogger(__name__)


def mask_redis_url(url: str) -> str:
    """Mask credentials in a Redis URL for safe logging."""
    if "@" in url and "://" in url:
        scheme, rest = url.split("://", 1)
        _, host_port_db = rest.split("@", 1)
        return f"{scheme}://***@{host_port_db}"
    return url


class MemoryStore(ABC):
    """Abstract base class for conversation memory stores."""

    @abstractmethod
    def load(self, session_id: str) -> List[Message]:
        """Load conversation history for a session."""
        pass

    @abstractmethod
    def save(self, session_id: str, messages: List[Message]) -> None:
        """Save conversation history for a session, trimming to k before storing."""
        pass

    @abstractmethod
    def clear(self, session_id: str) -> bool:
        """Clear memory for a session. Returns True if a session existed."""
        pass

    @abstractmethod
    def exists(self, session_id: str) -> bool:
        """Check whether memory exists for a session."""
        pass

    @abstractmethod
    def status(self) -> dict:
        """Return store status for health endpoints."""
        pass


class InProcessMemoryStore(MemoryStore):
    """In-process thread-safe dictionary memory store for local development and tests."""

    def __init__(self, k: int = 10):
        self.k = k
        self._storage: Dict[str, List[Message]] = {}
        self._lock = threading.Lock()

    def load(self, session_id: str) -> List[Message]:
        with self._lock:
            return list(self._storage.get(session_id, []))

    def save(self, session_id: str, messages: List[Message]) -> None:
        trimmed = codec.trim(messages, self.k)
        with self._lock:
            self._storage[session_id] = trimmed

    def clear(self, session_id: str) -> bool:
        with self._lock:
            return self._storage.pop(session_id, None) is not None

    def exists(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._storage

    def status(self) -> dict:
        return {
            "backend": "in_process",
            "connected": True,
            "circuit_open": False,
        }


class RedisMemoryStore(MemoryStore):
    """
    Redis-backed conversation memory store with sliding TTL, fail-open error handling,
    and a process-local circuit breaker.
    """

    def __init__(
        self,
        url: str,
        k: int = 10,
        ttl_seconds: int = 3600,
        key_prefix: str = "bsai:chat:session:",
        connect_timeout_s: float = 0.5,
        op_timeout_s: float = 0.5,
        circuit_breaker_s: int = 30,
    ):
        self.url = url
        self.k = k
        self.ttl_seconds = ttl_seconds
        self.key_prefix = key_prefix
        self.connect_timeout_s = connect_timeout_s
        self.op_timeout_s = op_timeout_s
        self.circuit_breaker_s = circuit_breaker_s

        self._client: Optional[redis.Redis] = None
        self._client_lock = threading.Lock()

        # Circuit breaker state
        self._breaker_lock = threading.Lock()
        self._last_failure_time: Optional[float] = None
        self._breaker_open: bool = False

    def _get_client(self) -> redis.Redis:
        """Lazily initialize Redis client so uvicorn worker processes don't share instances."""
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    self._client = redis.Redis.from_url(
                        self.url,
                        socket_connect_timeout=self.connect_timeout_s,
                        socket_timeout=self.op_timeout_s,
                        health_check_interval=30,
                    )
        return self._client

    def _key(self, session_id: str) -> str:
        return f"{self.key_prefix}{session_id}"

    def _is_circuit_open(self) -> bool:
        with self._breaker_lock:
            if not self._breaker_open:
                return False
            if self._last_failure_time is not None:
                if time.time() - self._last_failure_time < self.circuit_breaker_s:
                    return True
            # Cooldown passed; allow attempt
            return False

    def _handle_failure(self, error: Exception) -> None:
        with self._breaker_lock:
            now = time.time()
            if not self._breaker_open:
                self._breaker_open = True
                logger.warning(
                    "Circuit breaker OPENED for Redis memory store (cooldown %ds)",
                    self.circuit_breaker_s,
                )
            self._last_failure_time = now
        logger.warning(
            "Redis memory store operation failed: %s: %s (target: %s)",
            type(error).__name__,
            error,
            mask_redis_url(self.url),
        )

    def _handle_success(self) -> None:
        with self._breaker_lock:
            if self._breaker_open:
                self._breaker_open = False
                self._last_failure_time = None
                logger.info("Circuit breaker CLOSED for Redis memory store - connection restored")

    def load(self, session_id: str) -> List[Message]:
        if self._is_circuit_open():
            return []

        key = self._key(session_id)
        raw = None
        try:
            client = self._get_client()
            raw = client.get(key)
            self._handle_success()
        except (redis.RedisError, OSError) as e:
            self._handle_failure(e)
            return []

        if raw is None:
            return []

        raw_str = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        try:
            return codec.loads(raw_str)
        except Exception as e:
            logger.warning(
                "Corrupted session memory for key %s (%s). Deleting corrupted key.",
                key,
                e,
            )
            try:
                client.delete(key)
            except (redis.RedisError, OSError) as de:
                self._handle_failure(de)
            return []

    def save(self, session_id: str, messages: List[Message]) -> None:
        if self._is_circuit_open():
            return

        key = self._key(session_id)
        try:
            trimmed = codec.trim(messages, self.k)
            payload = codec.dumps(trimmed)
            client = self._get_client()
            client.set(key, payload, ex=self.ttl_seconds)
            self._handle_success()
        except (redis.RedisError, OSError) as e:
            self._handle_failure(e)
        except Exception as e:
            logger.warning("Failed to serialize or save messages: %s", e)

    def clear(self, session_id: str) -> bool:
        if self._is_circuit_open():
            return False

        key = self._key(session_id)
        try:
            client = self._get_client()
            deleted = client.delete(key)
            self._handle_success()
            return bool(deleted > 0)
        except (redis.RedisError, OSError) as e:
            self._handle_failure(e)
            return False

    def exists(self, session_id: str) -> bool:
        if self._is_circuit_open():
            return False

        key = self._key(session_id)
        try:
            client = self._get_client()
            res = client.exists(key)
            self._handle_success()
            return bool(res > 0)
        except (redis.RedisError, OSError) as e:
            self._handle_failure(e)
            return False

    def status(self) -> dict:
        if self._is_circuit_open():
            return {
                "backend": "redis",
                "connected": False,
                "circuit_open": True,
            }

        try:
            client = self._get_client()
            client.ping()
            self._handle_success()
            return {
                "backend": "redis",
                "connected": True,
                "circuit_open": False,
            }
        except (redis.RedisError, OSError) as e:
            self._handle_failure(e)
            return {
                "backend": "redis",
                "connected": False,
                "circuit_open": True,
            }


_memory_store_lock = threading.Lock()
_memory_store_instance: Optional[MemoryStore] = None


def _create_memory_store() -> MemoryStore:
    """Build a MemoryStore instance from application configuration."""
    from config import get_conversation_memory_config

    try:
        cfg = get_conversation_memory_config()
        backend = (getattr(cfg, "backend", "in_process") or "in_process").strip().lower()
        k = cfg.buffer_window.k
        ttl_minutes = getattr(cfg, "ttl_minutes", 60)
        ttl_seconds = ttl_minutes * 60
    except Exception as e:
        logger.error(
            "Failed to load conversation memory configuration: %s. Defaulting to in_process.",
            e,
        )
        return InProcessMemoryStore(k=10)

    if backend == "in_process":
        return InProcessMemoryStore(k=k)
    elif backend == "redis":
        redis_url = os.getenv("REDIS_URL")
        if not redis_url:
            logger.error("REDIS_URL environment variable is missing. Falling back to InProcessMemoryStore.")
            return InProcessMemoryStore(k=k)

        redis_cfg = getattr(cfg, "redis", None)
        key_prefix = getattr(redis_cfg, "key_prefix", "bsai:chat:session:")
        connect_timeout_s = getattr(redis_cfg, "connect_timeout_s", 0.5)
        op_timeout_s = getattr(redis_cfg, "op_timeout_s", 0.5)
        circuit_breaker_s = getattr(redis_cfg, "circuit_breaker_s", 30)

        return RedisMemoryStore(
            url=redis_url,
            k=k,
            ttl_seconds=ttl_seconds,
            key_prefix=key_prefix,
            connect_timeout_s=connect_timeout_s,
            op_timeout_s=op_timeout_s,
            circuit_breaker_s=circuit_breaker_s,
        )
    else:
        logger.error("Unknown conversation memory backend '%s'. Falling back to InProcessMemoryStore.", backend)
        return InProcessMemoryStore(k=k)


def get_memory_store() -> MemoryStore:
    """Get or create the process-wide MemoryStore singleton."""
    global _memory_store_instance
    if _memory_store_instance is None:
        with _memory_store_lock:
            if _memory_store_instance is None:
                _memory_store_instance = _create_memory_store()
    return _memory_store_instance


def _reset_memory_store() -> None:
    """Internal helper to reset singleton for testing."""
    global _memory_store_instance
    with _memory_store_lock:
        _memory_store_instance = None

