#!/usr/bin/env python3
"""
Verify V2 item 5 (instant greetings, 2-round cap, gatekeeper routing, metrics).

Part 1 checks the CODE on disk (no backend needed).
Part 2 checks the RUNNING backend (start it first). Uses about 4 Gemini calls:
"Hi"/"Thanks" use none, the out-of-scope message uses 1 gatekeeper call,
the real question uses 1 gatekeeper call + ~2-3 agent calls.

Run from backend/:
    python scripts/verify_item5.py
    python scripts/verify_item5.py --api http://localhost:8042 --hospital 62
    python scripts/verify_item5.py --offline        # Part 1 only
"""
import argparse
import hashlib
import json
import sqlite3
import sys
import time
import urllib.request
import uuid
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))
DB = BACKEND / "data" / "database" / "feedback.db"

results = []


def record(status, name, detail=""):
    results.append(status)
    print(f"  [{status:<4}] {name}" + (f"  - {detail}" if detail else ""))


def read(path):
    try:
        return (BACKEND / path).read_text(encoding="utf-8")
    except Exception:  # noqa: BLE001
        return None


# ── Part 1: code on disk ─────────────────────────────────────────────────────

def part1():
    print("Part 1 - code on disk")

    g = read("fast_path/greetings.py")
    record("PASS" if g and "def match_greeting" in g else "FAIL", "fast_path/greetings.py with match_greeting",
           "found" if g else "file missing")

    cfg = read("config.yaml") or ""
    import yaml
    try:
        c = yaml.safe_load(cfg) or {}
    except Exception as e:  # noqa: BLE001
        c = {}
        record("FAIL", "config.yaml parses", str(e))
    mi = (c.get("agent") or {}).get("max_iterations")
    record("PASS" if mi == 2 else "FAIL", "agent.max_iterations = 2", f"found {mi!r}")
    fp = c.get("fast_path") or {}
    replies = fp.get("replies") or {}
    record("PASS" if fp.get("enabled") is True and all(k in replies for k in ("greeting", "thanks", "bye", "out_of_scope"))
           else "FAIL", "fast_path enabled with 4 reply texts", f"enabled={fp.get('enabled')!r}, replies={sorted(replies)}")
    par = (c.get("performance") or {}).get("parallel_gatekeeper")
    record("PASS" if par is False else "WARN", "performance.parallel_gatekeeper = false", f"found {par!r}")

    ri = read("prompts/react_instructions.md") or ""
    record("PASS" if "SAME step" in ri else "FAIL", "Parallel-tools rule in prompts/react_instructions.md",
           "found" if "SAME step" in ri else "missing - the prompt version will not change")

    ra = read("agents/react_agent.py") or ""
    forced = "Step limit reached" in ra and "tool_choice" in ra
    record("PASS" if forced else "FAIL", "Forced final answer at the step limit (react_agent.py)",
           "found" if forced else "'Step limit reached' / tool_choice not found")

    base = read("llm/base.py") or ""
    record("PASS" if "cached_tokens" in base and "tool_choice" in base else "FAIL",
           "cached_tokens + tool_choice in llm/base.py")

    models = read("database/models.py") or ""
    record("PASS" if "request_metrics" in models else "FAIL", "RequestMetric model (request_metrics table)")

    if DB.exists():
        con = sqlite3.connect(DB)
        have = con.execute("select name from sqlite_master where name='request_metrics'").fetchall()
        record("PASS" if have else "WARN", "request_metrics table exists in feedback.db",
               "yes" if have else "not yet - it is created when the backend starts")
        con.close()
    else:
        record("WARN", "feedback.db found", f"not at {DB}")

    record("PASS" if (BACKEND / "tests" / "test_latency_cost.py").exists() else "WARN",
           "tests/test_latency_cost.py exists")


# ── Part 2: running backend ──────────────────────────────────────────────────

def get(api, path):
    with urllib.request.urlopen(f"{api}{path}", timeout=15) as r:
        return json.loads(r.read().decode("utf-8"))


def chat(api, message, session_id, hospital):
    body = {"message": message, "session_id": session_id}
    if hospital:
        body["hospital"] = hospital
    req = urllib.request.Request(f"{api}/api/chat/", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read().decode("utf-8"))
    return time.perf_counter() - t, data


def metric_for(message_id):
    if not DB.exists() or not message_id:
        return None
    try:
        con = sqlite3.connect(DB)
        con.row_factory = sqlite3.Row
        row = con.execute("select * from request_metrics where message_id = ? order by timestamp desc limit 1",
                          (message_id,)).fetchone()
        con.close()
        return dict(row) if row else None
    except Exception:  # noqa: BLE001
        return None


def part2(api, hospital):
    print(f"\nPart 2 - running backend ({api})")
    try:
        h = get(api, "/api/health")
    except Exception as e:  # noqa: BLE001
        record("FAIL", "Backend reachable", f"{type(e).__name__}: {e} - start it first")
        return

    # Is the running backend using the prompt files on disk?
    try:
        from prompts.loader import prompt_version
        disk = prompt_version()
    except Exception:  # noqa: BLE001
        disk = None
    live = h.get("prompt_version")
    if disk and live:
        record("PASS" if disk == live else "FAIL", "Backend runs the current prompt files",
               f"live {live} = disk {disk}" if disk == live else
               f"live {live} != disk {disk} - RESTART the backend")
    else:
        record("WARN", "Prompt version check", f"live={live!r}, disk={disk!r}")

    run = uuid.uuid4().hex[:8]
    cfg = {}
    try:
        import yaml
        cfg = (yaml.safe_load(read("config.yaml") or "") or {}).get("fast_path", {}).get("replies", {})
    except Exception:  # noqa: BLE001
        pass

    cases = [
        ("Hi", "fast_path", cfg.get("greeting"), 1.0),
        ("Thanks!", "fast_path", cfg.get("thanks"), 1.0),
        ("Write me a poem about the ocean", "guardrail_route", cfg.get("out_of_scope"), 15.0),
    ]
    for msg, want_path, want_reply, max_s in cases:
        try:
            secs, d = chat(api, msg, f"verify5_{run}_{abs(hash(msg)) % 10000}", hospital)
        except Exception as e:  # noqa: BLE001
            record("FAIL", f"{msg!r}", f"{type(e).__name__}: {e}")
            continue
        reply = (d.get("response") or "").strip()
        ok_reply = want_reply is None or reply == want_reply.strip()
        ok_fast = secs <= max_s
        ok_provider = d.get("provider_used") == want_path
        status = "PASS" if (ok_reply and ok_provider and ok_fast) else "FAIL"
        record(status, f"{msg!r} -> {want_path}",
               f"{secs:.2f}s, provider_used={d.get('provider_used')!r}"
               + ("" if ok_reply else f", reply={reply[:70]!r}"))
        m = metric_for(d.get("message_id"))
        if m:
            record("PASS" if m.get("path") == want_path else "FAIL", "   metric row",
                   f"path={m.get('path')}, llm_calls={m.get('llm_calls')}, total_ms={m.get('total_ms')}")
        else:
            record("FAIL", "   metric row", "no request_metrics row for this message")

    try:
        secs, d = chat(api, "How do I upload a scan?", f"verify5_{run}_real", hospital)
        rounds = len(d.get("react_steps") or [])
        record("PASS" if d.get("provider_used") not in ("fast_path", "guardrail_route", "guardrail")
               and d.get("response") else "FAIL", "Real question reaches the agent",
               f"{secs:.1f}s, provider_used={d.get('provider_used')!r}")
        m = metric_for(d.get("message_id"))
        if m:
            ok = m.get("path") in ("agent", "agent_capped") and (m.get("agent_rounds") or 0) <= 3
            record("PASS" if ok else "FAIL", "   metric row (agent, capped at 2 rounds + final)",
                   f"path={m.get('path')}, agent_rounds={m.get('agent_rounds')}, llm_calls={m.get('llm_calls')}, "
                   f"tokens={m.get('tokens')}, cached={m.get('cached_tokens')}, total_ms={m.get('total_ms')}")
        else:
            record("FAIL", "   metric row", "no request_metrics row for this message")
            record("INFO", "   steps in response", str(rounds))
    except Exception as e:  # noqa: BLE001
        record("FAIL", "Real question", f"{type(e).__name__}: {e}")


def main():
    try:
        from dotenv import load_dotenv
        load_dotenv(BACKEND / ".env")
    except ImportError:
        pass
    ap = argparse.ArgumentParser(description="Verify V2 item 5")
    ap.add_argument("--api", default="http://localhost:8042")
    ap.add_argument("--hospital", default="62")
    ap.add_argument("--offline", action="store_true", help="only check the code on disk")
    args = ap.parse_args()

    print("\nBSAI item 5 verification\n")
    part1()
    if not args.offline:
        part2(args.api.rstrip("/"), args.hospital)

    fails = results.count("FAIL")
    print("\n" + "-" * 60)
    print(f"PASS {results.count('PASS')}   FAIL {fails}   WARN {results.count('WARN')}   INFO {results.count('INFO')}")
    print("Item 5 is in place and working." if not fails else "Item 5 is incomplete - see the FAIL lines.")
    return 1 if fails else 0



if __name__ == "__main__":
    sys.exit(main())