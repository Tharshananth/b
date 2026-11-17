#!/usr/bin/env python3
"""
Pretest for V2 item 3 (conversation memory in Redis).

Checks that the library versions, the Redis server, and the planned design
(serialize -> trim -> save with TTL -> load from any worker) all work with
OUR engine's real Message / ToolCall classes, before touching chat.py.

It changes nothing in the app. It only writes keys under
`bsai:pretest:<run-id>:` and deletes them at the end.

Run from backend/:
    python scripts/pretest_redis_memory.py
    python scripts/pretest_redis_memory.py --url redis://:password@host:6379/0
    python scripts/pretest_redis_memory.py --url rediss://:key@name.redis.cache.windows.net:6380/0   (Azure)

The URL can also come from REDIS_URL in the environment / .env.
Exit code is 0 if nothing FAILed, 1 otherwise.
"""
import argparse
import base64
import importlib.util
import json
import multiprocessing as mp
import os
import statistics
import sys
import threading
import time
import uuid
from dataclasses import fields
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
EXPECTED_REDIS_PY = "5.0.1"          # pinned in requirement.txt
TRIM_K = 10                          # config.yaml conversation_memory.buffer_window.k

RUN_ID = uuid.uuid4().hex[:8]
PREFIX = f"bsai:pretest:{RUN_ID}:"

results = []  # (status, name, detail)


def record(status, name, detail=""):
    results.append((status, name, detail))
    colour = {"PASS": "32", "FAIL": "31", "WARN": "33", "INFO": "36"}.get(status, "0")
    use_colour = sys.stdout.isatty() and os.name != "nt"
    tag = f"\033[{colour}m{status:<4}\033[0m" if use_colour else f"{status:<4}"
    print(f"  [{tag}] {name}" + (f"  - {detail}" if detail else ""))


def check(name, fn):
    """Run one check; any exception is a FAIL with the reason."""
    try:
        out = fn()
        if isinstance(out, tuple):
            record(*out[:1], name, *out[1:])
        else:
            record("PASS", name, out or "")
    except AssertionError as e:
        record("FAIL", name, str(e) or "assertion failed")
    except Exception as e:  # noqa: BLE001
        record("FAIL", name, f"{type(e).__name__}: {e}")


# ── Load our real Message / ToolCall without importing the provider SDKs ─────
# llm/__init__.py imports every provider (openai, anthropic, google...), so we
# load llm/base.py directly from its file.

def load_engine_classes():
    spec = importlib.util.spec_from_file_location("bsai_llm_base", BACKEND / "llm" / "base.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Message, mod.ToolCall


Message, ToolCall = load_engine_classes()


# ── The codec we plan to use (bytes-safe JSON) ───────────────────────────────

def _to_jsonable(obj):
    if isinstance(obj, bytes):
        return {"__bytes__": base64.b64encode(obj).decode("ascii")}
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def _from_jsonable(obj):
    if isinstance(obj, dict):
        if set(obj) == {"__bytes__"}:
            return base64.b64decode(obj["__bytes__"])
        return {k: _from_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_from_jsonable(v) for v in obj]
    return obj


def message_to_dict(m):
    d = {f.name: getattr(m, f.name) for f in fields(m)}
    if m.tool_calls:
        d["tool_calls"] = [{f.name: getattr(tc, f.name) for f in fields(tc)} for tc in m.tool_calls]
    return _to_jsonable(d)


def message_from_dict(d):
    d = _from_jsonable(d)
    tcs = d.get("tool_calls")
    if tcs:
        d["tool_calls"] = [ToolCall(**tc) for tc in tcs]
    return Message(**d)


def dumps(messages):
    return json.dumps([message_to_dict(m) for m in messages], ensure_ascii=False)


def loads(raw):
    return [message_from_dict(d) for d in json.loads(raw)]


# ── Trim: same rule as chat.py _load_history, applied before saving ──────────

def trim(messages, k=TRIM_K):
    starts = [i for i, m in enumerate(messages) if m.role == "user"]
    return messages[starts[-k]:] if len(starts) > k else messages


# ── Sample conversation that looks like what our ReAct agent stores ──────────

def tool_exchange(turn, session="s"):
    """One user turn with a tool call, as the agent stores it (4 messages)."""
    call_id = f"call_{session}_{turn}"
    return [
        Message(role="user", content=f"[{session}] turn {turn}: status of patient 2013?"),
        Message(role="assistant", content="", tool_calls=[ToolCall(
            id=call_id, name="get_job_status",
            arguments={"patient_id": "2013", "include_stages": True, "turn": turn},
            # Gemini puts raw bytes here and rejects the next turn without them.
            provider_data={"thought_signature": bytes([turn % 256, 0, 255, 17]) + b"\x00sig"},
        )]),
        Message(role="tool", content=json.dumps({"status": "failed", "stage": "registration"}),
                tool_call_id=call_id, name="get_job_status"),
        Message(role="assistant", content=f"[{session}] turn {turn}: it failed at registration."),
    ]


def plain_turn(turn, session="s"):
    return [Message(role="user", content=f"[{session}] hi {turn}"),
            Message(role="assistant", content=f"[{session}] hello {turn}")]


# ── Worker-process helpers (top level so Windows 'spawn' can pickle them) ────

def _proc_write(url, key, payload):
    import redis
    redis.Redis.from_url(url, socket_timeout=3).set(key, payload, ex=60)


def _proc_read(url, key):
    import redis
    v = redis.Redis.from_url(url, socket_timeout=3).get(key)
    return v.decode("utf-8") if v is not None else None


def _proc_turn(args):
    """One chat turn handled by whichever worker process picked it up."""
    url, prefix, session, turn = args
    import redis
    r = redis.Redis.from_url(url, socket_timeout=3)
    key = f"{prefix}session:{session}"
    raw = r.get(key)
    history = loads(raw.decode("utf-8")) if raw else []
    history += tool_exchange(turn, session) if turn % 2 else plain_turn(turn, session)
    r.set(key, dumps(trim(history)), ex=60)
    return os.getpid()


# ── The checks ───────────────────────────────────────────────────────────────

def run(url, workers):
    import redis

    print(f"\nBSAI Redis memory pretest  (run id {RUN_ID})")
    print(f"Target: {redact(url)}\n")

    # 1. Environment
    print("Environment")
    check("Python version", lambda: (("PASS" if sys.version_info >= (3, 9) else "FAIL"),
                                     sys.version.split()[0]))
    check("redis-py version",
          lambda: ("PASS", redis.__version__) if redis.__version__ == EXPECTED_REDIS_PY
          else ("WARN", f"{redis.__version__} installed, requirement.txt pins {EXPECTED_REDIS_PY}"))
    check("Engine Message/ToolCall classes load",
          lambda: f"Message fields: {[f.name for f in fields(Message)]}")

    # 2. Connection
    print("\nConnection")
    r = redis.Redis.from_url(url, socket_connect_timeout=2, socket_timeout=2)
    try:
        r.ping()
    except Exception as e:  # noqa: BLE001
        record("FAIL", "Connect + PING", f"{type(e).__name__}: {e}")
        print("\nCannot reach Redis - remaining checks skipped. See tips at the bottom.")
        return summary(connect_failed=True)

    def server_info():
        info = r.info("server")
        v = info.get("redis_version", "?")
        major = int(v.split(".")[0]) if v[0].isdigit() else 0
        return ("PASS" if major >= 6 else "WARN", f"Redis server {v}")
    check("Connect + PING", lambda: "ok")
    check("Server version (6+ recommended)", server_info)

    def ping_latency():
        ts = []
        for _ in range(50):
            t = time.perf_counter(); r.ping(); ts.append((time.perf_counter() - t) * 1000)
        p50 = statistics.median(ts)
        return ("PASS" if p50 < 20 else "WARN", f"p50 {p50:.2f} ms")
    check("Round-trip latency", ping_latency)

    # 3. Basic operations
    print("\nBasic operations")

    def basic():
        k = PREFIX + "basic"
        r.set(k, "தமிழ் + emoji 🧠 ok", ex=30)
        assert r.get(k).decode("utf-8") == "தமிழ் + emoji 🧠 ok", "unicode value changed"
        r.delete(k)
        assert r.get(k) is None, "delete did not remove key"
    check("SET / GET / DEL with unicode", basic)

    # 4. Serialization with our real classes
    print("\nSerialization (our Message / ToolCall)")
    convo = tool_exchange(1) + plain_turn(2) + tool_exchange(3)

    def naive_json_fails():
        try:
            json.dumps([vars(m) for m in convo], default=lambda o: vars(o))
            return ("WARN", "plain json.dumps unexpectedly succeeded")
        except TypeError:
            return ("PASS", "confirmed: plain JSON crashes on Gemini's bytes -> custom codec needed")
    check("Plain json.dumps is unsafe", naive_json_fails)

    def to_dict_lossy():
        d = convo[1].to_dict()
        return ("PASS" if "tool_calls" not in d else "WARN",
                "confirmed: Message.to_dict() drops tool_calls -> cannot reuse it for memory")
    check("Existing Message.to_dict() is lossy", to_dict_lossy)

    def roundtrip():
        back = loads(dumps(convo))
        assert back == convo, "messages differ after round trip"
        sig = back[1].tool_calls[0].provider_data["thought_signature"]
        assert isinstance(sig, bytes), f"thought_signature came back as {type(sig).__name__}, not bytes"
        return "all fields equal, bytes preserved"
    check("Codec round trip (in memory)", roundtrip)

    def roundtrip_redis():
        k = PREFIX + "roundtrip"
        r.set(k, dumps(convo), ex=30)
        back = loads(r.get(k).decode("utf-8"))
        assert back == convo, "messages differ after Redis round trip"
        return f"{len(convo)} messages through Redis unchanged"
    check("Codec round trip (through Redis)", roundtrip_redis)

    # 5. Trimming
    print("\nTrimming (k = %d user turns)" % TRIM_K)

    def trimming():
        long = []
        for t in range(1, 16):
            long += tool_exchange(t) if t % 2 else plain_turn(t)
        kept = trim(long)
        users = [m for m in kept if m.role == "user"]
        assert len(users) == TRIM_K, f"kept {len(users)} user turns"
        assert kept[0].role == "user", "trimmed history doesn't start at a user turn"
        call_ids = {tc.id for m in kept if m.tool_calls for tc in m.tool_calls}
        results_ids = {m.tool_call_id for m in kept if m.role == "tool"}
        assert call_ids == results_ids, "a tool call lost its result (or vice versa)"
        return f"{len(long)} -> {len(kept)} messages, no dangling tool calls"
    check("Trim keeps last k turns with tool turns intact", trimming)

    # 6. TTL
    print("\nExpiry (TTL) - takes ~7 s")

    def ttl_basic():
        k = PREFIX + "ttl"
        r.set(k, "x", ex=2)
        assert r.ttl(k) in (1, 2), f"TTL is {r.ttl(k)}"
        time.sleep(3)
        assert r.get(k) is None, "key still exists after expiry"
        return "key gone after 2 s"
    check("Key expires after TTL", ttl_basic)

    def ttl_sliding():
        k = PREFIX + "slide"
        r.set(k, "turn1", ex=3)
        time.sleep(2)
        r.set(k, "turn2", ex=3)        # a new message resets the clock
        time.sleep(2)
        assert r.get(k) == b"turn2", "session expired although it was active"
        time.sleep(2)
        assert r.get(k) is None, "session did not expire after going idle"
        return "active session kept alive, idle session expired"
    check("Sliding expiry (each save resets TTL)", ttl_sliding)

    # 7. Multiple workers (separate processes, like uvicorn --workers)
    print(f"\nMultiple workers ({workers} processes)")
    ctx = mp.get_context("spawn")      # same start method as Windows

    def cross_process():
        k = PREFIX + "xproc"
        payload = dumps(convo)
        p = ctx.Process(target=_proc_write, args=(url, k, payload)); p.start(); p.join(20)
        assert p.exitcode == 0, "writer process failed"
        with ctx.Pool(1) as pool:
            got = pool.apply(_proc_read, (url, k))
        assert got is not None, "reader process saw nothing"
        assert loads(got) == convo, "reader process got different data"
        return "process A wrote, process B read the same conversation"
    check("Worker A writes, worker B reads", cross_process)

    def ten_users():
        users, turns = 10, 5
        pids = set()
        with ctx.Pool(workers) as pool:
            for turn in range(1, turns + 1):          # each round = everyone sends a message
                jobs = [(url, PREFIX, f"user{u}", turn) for u in range(users)]
                pids.update(pool.map(_proc_turn, jobs, chunksize=1))
        for u in range(users):
            raw = r.get(f"{PREFIX}session:user{u}")
            assert raw, f"user{u} has no session"
            hist = loads(raw.decode("utf-8"))
            user_msgs = [m.content for m in hist if m.role == "user"]
            assert len(user_msgs) == turns, f"user{u} has {len(user_msgs)} turns, expected {turns}"
            assert all(c.startswith(f"[user{u}]") for c in user_msgs), f"user{u} sees another user's messages"
            assert [int(c.split("turn " if "turn" in c else "hi ")[1].split(":")[0]) for c in user_msgs] \
                == list(range(1, turns + 1)), f"user{u} turns out of order"
        return f"{users} users x {turns} turns over {len(pids)} worker processes: no lost turns, no mixing"
    check("10 concurrent users, turns land on random workers", ten_users)

    def double_click():
        k = PREFIX + "race"
        r.set(k, dumps(plain_turn(1)), ex=30)
        barrier = threading.Barrier(2)

        def send(turn):
            hist = loads(r.get(k).decode("utf-8"))
            barrier.wait()                     # both read the same state first
            r.set(k, dumps(hist + plain_turn(turn)), ex=30)
        ts = [threading.Thread(target=send, args=(n,)) for n in (2, 3)]
        [t.start() for t in ts]; [t.join() for t in ts]
        n = len([m for m in loads(r.get(k).decode("utf-8")) if m.role == "user"])
        return ("INFO", f"same session sent twice at once -> {n} of 3 turns kept "
                        "(last write wins; acceptable, documented)")
    check("Same user double-sends (race)", double_click)

    # 8. Failure handling
    print("\nFailure handling")

    def fail_open():
        dead = redis.Redis(host="127.0.0.1", port=1, socket_connect_timeout=1, socket_timeout=1)

        def load_or_empty(session):    # the planned fail-open behaviour
            try:
                raw = dead.get(PREFIX + session)
                return loads(raw.decode("utf-8")) if raw else []
            except redis.RedisError:
                return []
        t = time.perf_counter()
        hist = load_or_empty("x")
        took = time.perf_counter() - t
        assert hist == [], "fail-open did not return empty history"
        assert took < 3, f"took {took:.1f} s - timeouts too long, chat would stall"
        return f"Redis unreachable -> empty history in {took*1000:.0f} ms, chat continues"
    check("Redis down -> fail open quickly", fail_open)

    # 9. Size & performance
    print("\nSize & performance")

    def size():
        full = []
        for t in range(1, TRIM_K + 1):
            full += tool_exchange(t)
        kb = len(dumps(full).encode("utf-8")) / 1024
        return ("PASS" if kb < 512 else "WARN", f"full 10-turn tool-heavy session = {kb:.1f} KB")
    check("Session payload size", size)

    def perf():
        k = PREFIX + "perf"
        hist = []
        for t in range(1, TRIM_K + 1):
            hist += tool_exchange(t)
        payload = dumps(hist)
        ts = []
        for _ in range(300):
            t = time.perf_counter()
            r.set(k, payload, ex=60)
            loads(r.get(k).decode("utf-8"))
            ts.append((time.perf_counter() - t) * 1000)
        ts.sort()
        p50, p95 = ts[len(ts) // 2], ts[int(len(ts) * 0.95)]
        return ("PASS" if p95 < 50 else "WARN", f"save+load p50 {p50:.2f} ms, p95 {p95:.2f} ms")
    check("Save + load latency (300 ops)", perf)

    return summary()


def cleanup(url):
    try:
        import redis
        r = redis.Redis.from_url(url, socket_connect_timeout=2, socket_timeout=2)
        n = 0
        for key in r.scan_iter(match=PREFIX + "*", count=500):
            r.delete(key); n += 1
        print(f"\nCleanup: removed {n} pretest keys (prefix {PREFIX})")
    except Exception:  # noqa: BLE001
        pass


def redact(url):
    if "@" in url and "://" in url:
        scheme, rest = url.split("://", 1)
        return f"{scheme}://***@{rest.split('@', 1)[1]}"
    return url


def summary(connect_failed=False):
    counts = {s: sum(1 for r in results if r[0] == s) for s in ("PASS", "FAIL", "WARN", "INFO")}
    print("\n" + "-" * 60)
    print(f"PASS {counts['PASS']}   FAIL {counts['FAIL']}   WARN {counts['WARN']}   INFO {counts['INFO']}")
    if counts["FAIL"] == 0:
        print("Ready: the Redis memory design works with this engine and server.")
    else:
        print("Not ready: fix the FAILed checks above before implementing.")
    if connect_failed:
        print("\nTips:")
        print("  Local (Docker):  docker run -d --name bsai-redis -p 6379:6379 redis:7")
        print("  Windows without Docker: use WSL ('sudo apt install redis-server') or Memurai.")
        print("  Azure Cache for Redis: use rediss:// (TLS) on port 6380 with the access key.")
    return 1 if counts["FAIL"] else 0


def main():
    try:
        from dotenv import load_dotenv
        load_dotenv(BACKEND / ".env")
    except ImportError:
        pass
    ap = argparse.ArgumentParser(description="Pretest Redis conversation memory for the BSAI chatbot")
    ap.add_argument("--url", default=os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    ap.add_argument("--workers", type=int, default=4, help="processes to simulate (uvicorn workers)")
    args = ap.parse_args()
    try:
        code = run(args.url, args.workers)
    finally:
        cleanup(args.url)
    sys.exit(code)


if __name__ == "__main__":
    main()