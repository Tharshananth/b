#!/usr/bin/env python3
"""
Pretest for V2 item 5 (latency & cost).

Five parts, each can be run on its own:

  A  baseline   Calls YOUR running backend (/api/chat/) with greetings, out-of-scope,
                medical, technical and data questions. Records time, LLM rounds,
                tools and tokens per message type. Saves a JSON "before" file;
                run again after the changes with --compare to see the improvement.
  B  greetings  Offline test of the instant-greeting matcher (no AI, no network).
  C  caching    Calls Gemini directly with our real system prompt several times and
                checks whether implicit prompt caching kicks in (cached tokens > 0).
  D  final      Checks the "limit reached -> answer with what you have" call:
                a real tool-call history + tools switched OFF must return text.
  E  parallel   Checks whether the model asks for several tools in ONE step.

Run from backend/:
    python scripts/pretest_latency_cost.py                       # all parts
    python scripts/pretest_latency_cost.py --parts B             # offline only
    python scripts/pretest_latency_cost.py --parts A --api http://localhost:8042
    python scripts/pretest_latency_cost.py --parts A --compare reports/baseline_XXXX.json
    python scripts/pretest_latency_cost.py --parts CDE --model gemini-2.5-flash

Part A needs the backend running. Parts C-E need GOOGLE_API_KEY.
It changes nothing in the app. Exit code 1 if anything FAILed.
"""
import argparse
import importlib.util
import json
import os
import re
import statistics
import sys
import time
import unicodedata
import urllib.error
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

results = []


def record(status, name, detail=""):
    results.append(status)
    use_colour = sys.stdout.isatty() and os.name != "nt"
    colour = {"PASS": "32", "FAIL": "31", "WARN": "33", "INFO": "36", "SKIP": "90"}.get(status, "0")
    tag = f"\033[{colour}m{status:<4}\033[0m" if use_colour else f"{status:<4}"
    print(f"  [{tag}] {name}" + (f"  - {detail}" if detail else ""))


# ═════════════════════════════════════════════════════════════════════════════
# B. Instant-greeting matcher (candidate implementation)
# ═════════════════════════════════════════════════════════════════════════════
# The WHOLE message must be a greeting / thanks / goodbye. "ok", "yes", "sure"
# alone are NOT greetings: they often answer a question the bot just asked
# ("Shall I check the job?" -> "ok"), so they must reach the agent with context.

GREETING_WORDS = {"hi", "hii", "hiii", "hello", "helo", "hey", "heya", "hola",
                  "morning", "afternoon", "evening", "vanakkam", "namaste", "namaskar"}
THANKS_WORDS = {"thanks", "thank", "thx", "ty", "tq", "nandri", "dhanyavad", "dhanyawad",
                "shukriya", "appreciated"}
BYE_WORDS = {"bye", "goodbye", "cya", "tata"}
FILLER_WORDS = {"good", "you", "so", "much", "a", "lot", "very", "there", "all", "team",
                "again", "ok", "okay", "got", "it", "great", "cool", "nice", "and", "for",
                "the", "help", "your", "sir", "madam", "bot", "everyone", "see", "later",
                "ji", "romba", "bahut"}

GREETING_REPLY = "Hello! I can help with VoxelBox processing, MRI protocols, and job status. What do you need?"
THANKS_REPLY = "You're welcome! Let me know if you need anything else with VoxelBox."
BYE_REPLY = "Goodbye! Come back anytime you need help with VoxelBox."


def normalize(text: str) -> list:
    text = unicodedata.normalize("NFKC", text).lower()
    text = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in text)
    return text.split()


def match_greeting(text: str):
    """Return 'greeting' | 'thanks' | 'bye' if the whole message is one, else None."""
    words = normalize(text)
    if not words or len(words) > 6:
        return None
    core = set(words) & (GREETING_WORDS | THANKS_WORDS | BYE_WORDS)
    if not core:
        return None
    if any(w not in GREETING_WORDS | THANKS_WORDS | BYE_WORDS | FILLER_WORDS for w in words):
        return None
    if core & THANKS_WORDS:
        return "thanks"
    if core & BYE_WORDS:
        return "bye"
    return "greeting"


GREETING_CASES = [
    # (message, expected kind or None)
    ("Hi", "greeting"), ("hello!", "greeting"), ("Hey there", "greeting"), ("hii", "greeting"),
    ("Good morning", "greeting"), ("good evening team", "greeting"), ("Vanakkam", "greeting"),
    ("Namaste ji", "greeting"), ("Hello 👋", "greeting"), ("HI!!!", "greeting"),
    ("Thanks", "thanks"), ("thank you so much", "thanks"), ("Thanks a lot!", "thanks"),
    ("ok thanks", "thanks"), ("got it, thanks", "thanks"), ("thx", "thanks"), ("Nandri", "thanks"),
    ("romba nandri", "thanks"), ("bahut shukriya", "thanks"), ("great, thank you", "thanks"),
    ("bye", "bye"), ("Goodbye!", "bye"), ("ok bye", "bye"), ("see you later bye", "bye"),
    # must NOT match - go to gatekeeper / agent
    ("hi, what's the status of patient 2013?", None),
    ("hello can you help me upload a scan", None),
    ("thanks, but the upload failed", None),
    ("hi is my tumor back", None),
    ("ok", None), ("okay", None), ("yes", None), ("sure", None), ("no", None),
    ("ok do it", None), ("help", None), ("👍", None), ("", None),
    ("hi hi hi hi hi hi hi hi", None),
    ("thank you for nothing, the job failed again", None),
    ("good", None), ("great", None),
]


def part_b():
    print("\nB. Instant-greeting matcher (offline)")
    wrong = []
    for msg, expected in GREETING_CASES:
        got = match_greeting(msg)
        if got != expected:
            wrong.append((msg, expected, got))
    if wrong:
        record("FAIL", f"Greeting matcher ({len(GREETING_CASES) - len(wrong)}/{len(GREETING_CASES)})")
        for msg, exp, got in wrong:
            print(f"      {msg!r}: expected {exp}, got {got}")
    else:
        record("PASS", "Greeting matcher", f"{len(GREETING_CASES)}/{len(GREETING_CASES)} cases "
               "(greetings/thanks/bye caught; questions, 'ok', 'yes' left for the agent)")
    t = time.perf_counter()
    for _ in range(10000):
        match_greeting("thank you so much")
    us = (time.perf_counter() - t) / 10000 * 1e6
    record("PASS" if us < 1000 else "WARN", "Matcher speed", f"{us:.1f} µs per message")


# ═════════════════════════════════════════════════════════════════════════════
# A. Baseline against the running backend
# ═════════════════════════════════════════════════════════════════════════════

BASELINE_MESSAGES = [
    ("greeting", "Hi"),
    ("greeting", "Thank you!"),
    ("greeting", "Good morning"),
    ("out_of_scope", "Write me a short poem about the ocean."),
    ("out_of_scope", "What's the weather in Chennai?"),
    ("medical", "I have a headache, is my tumor back?"),
    ("medical", "What dose of dexamethasone should I take?"),
    ("technical", "What b-value does the DTI acquisition protocol use?"),
    ("technical", "How do I upload a scan to VoxelBox?"),
    ("technical", "What are the installation prerequisites for VoxelBox?"),
    ("data_lookup", "Which jobs are stuck in the registration stage?"),
    ("data_lookup", "How many scans failed this week?"),
]


def post_chat(api, message, hospital, timeout=120):
    body = {"message": message, "session_id": f"pretest_{uuid.uuid4().hex[:10]}"}
    if hospital:
        body["hospital"] = hospital
    req = urllib.request.Request(f"{api.rstrip('/')}/api/chat/", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return (time.perf_counter() - t), data


def part_a(api, hospital, repeat, compare):
    print(f"\nA. Baseline against the running backend ({api})")
    try:
        urllib.request.urlopen(f"{api.rstrip('/')}/health", timeout=5)
    except Exception as e:  # noqa: BLE001
        record("SKIP", "Backend reachable", f"{type(e).__name__}: {e} - start it, or pass --api")
        return
    record("PASS", "Backend reachable")

    rows = []
    for kind, msg in BASELINE_MESSAGES:
        for _ in range(repeat):
            try:
                secs, data = post_chat(api, msg, hospital)
                rows.append(dict(kind=kind, message=msg, seconds=round(secs, 2),
                                 rounds=len(data.get("react_steps") or []),
                                 tools=len(data.get("tools_used") or []),
                                 tokens=data.get("tokens_used") or 0,
                                 provider=data.get("provider_used"),
                                 path=data.get("guardrail") or data.get("provider_used"),
                                 success=data.get("success", True), error=None))
                print(f"      {secs:5.1f}s  rounds={rows[-1]['rounds']}  tools={rows[-1]['tools']}  "
                      f"tokens={rows[-1]['tokens']:>6}  [{kind}] {msg[:50]}")
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:150]
                rows.append(dict(kind=kind, message=msg, seconds=None, error=f"HTTP {e.code}: {detail}"))
                print(f"      ERROR  [{kind}] {msg[:50]} -> HTTP {e.code}: {detail}")
            except Exception as e:  # noqa: BLE001
                rows.append(dict(kind=kind, message=msg, seconds=None, error=f"{type(e).__name__}: {e}"))
                print(f"      ERROR  [{kind}] {msg[:50]} -> {type(e).__name__}: {e}")

    summary = {}
    for kind in dict.fromkeys(k for k, _ in BASELINE_MESSAGES):
        ok = [r for r in rows if r["kind"] == kind and r["seconds"] is not None]
        if ok:
            summary[kind] = dict(avg_s=round(statistics.mean(r["seconds"] for r in ok), 2),
                                 max_s=round(max(r["seconds"] for r in ok), 2),
                                 avg_rounds=round(statistics.mean(r["rounds"] for r in ok), 1),
                                 avg_tokens=round(statistics.mean(r["tokens"] for r in ok)))
    print("\n      per type:   avg time   max time   avg rounds   avg tokens")
    for kind, s in summary.items():
        print(f"      {kind:<12} {s['avg_s']:>7.2f}s  {s['max_s']:>8.2f}s  {s['avg_rounds']:>10}  {s['avg_tokens']:>11}")

    errors = [r for r in rows if r["error"]]
    record("PASS" if not errors else "WARN", "Baseline calls", f"{len(rows) - len(errors)}/{len(rows)} succeeded"
           + (" (data_lookup errors usually mean no hospital scope: try --hospital 62)" if errors else ""))
    if "greeting" in summary:
        g = summary["greeting"]["avg_s"]
        record("INFO", "Greeting time today", f"{g:.2f}s average (target after item 5: ~0s)")

    out_dir = BACKEND / "reports"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"baseline_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(dict(api=api, when=datetime.now().isoformat(), summary=summary, rows=rows),
                              indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"      saved: {out}")

    if compare:
        before = json.loads(Path(compare).read_text(encoding="utf-8"))["summary"]
        print(f"\n      comparison with {compare}:")
        print("      type          before    after    change")
        for kind, s in summary.items():
            if kind in before:
                b, a = before[kind]["avg_s"], s["avg_s"]
                pct = (a - b) / b * 100 if b else 0
                print(f"      {kind:<12} {b:>7.2f}s {a:>7.2f}s  {pct:+6.0f}%")


# ═════════════════════════════════════════════════════════════════════════════
# Gemini helpers for C, D, E
# ═════════════════════════════════════════════════════════════════════════════

def gemini_client():
    try:
        import google.genai as genai
        from google.genai import types
    except ImportError:
        record("FAIL", "google-genai installed", "pip install -U google-genai")
        return None, None
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        record("FAIL", "API key", "GOOGLE_API_KEY not set (backend/.env)")
        return None, None
    return genai.Client(api_key=key), types


def real_system_prompt():
    """Build the agent's system prompt exactly as the app does, without
    importing the agents/llm packages (their __init__ pulls in every SDK)."""
    spec = importlib.util.spec_from_file_location("bsai_agent_prompt", BACKEND / "agents" / "prompt.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_react_system_prompt()


def tool_decls(types):
    """Two simple tools shaped like ours (enough to test the call patterns)."""
    return [types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="get_job_status",
            description="Get the processing status and failed stage of a patient's VoxelBox job.",
            parameters=types.Schema(type="OBJECT", properties={
                "patient_id": types.Schema(type="STRING", description="Patient ID")},
                required=["patient_id"])),
        types.FunctionDeclaration(
            name="search_knowledge_base",
            description="Search BrainSightAI manuals and protocols.",
            parameters=types.Schema(type="OBJECT", properties={
                "query": types.Schema(type="STRING", description="What to search for")},
                required=["query"])),
    ])]


def base_cfg(types, system, tools=None, mode=None):
    kw = dict(system_instruction=system, temperature=0.2, max_output_tokens=600,
              automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
    if tools:
        kw["tools"] = tools
    if mode:
        kw["tool_config"] = types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode=mode))
    return types.GenerateContentConfig(**kw)


def function_calls(resp):
    out = []
    for cand in (resp.candidates or [])[:1]:
        for part in (cand.content.parts or []) if cand.content else []:
            if getattr(part, "function_call", None):
                out.append(part)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# C. Prompt caching
# ═════════════════════════════════════════════════════════════════════════════

def part_c(client, types, model, calls):
    print(f"\nC. Prompt caching with our real system prompt ({model})")
    try:
        system = real_system_prompt()
    except Exception as e:  # noqa: BLE001
        record("FAIL", "Build real system prompt", f"{type(e).__name__}: {e}")
        return
    again = real_system_prompt()
    record("PASS" if system == again else "FAIL", "System prompt identical on every build",
           "stable prefix -> cacheable" if system == again else "changes between calls -> caching will miss")

    questions = ["What b-value does the DTI protocol use?", "How do I upload a scan?",
                 "What GPU does VoxelBox need?", "What does motion correction do?",
                 "Which DICOM formats are supported?", "How long does tractography take?"]
    rows = []
    for i in range(calls):
        q = questions[i % len(questions)]
        t = time.perf_counter()
        try:
            resp = client.models.generate_content(
                model=model, contents=[types.Content(role="user", parts=[types.Part(text=q)])],
                config=base_cfg(types, system, tools=tool_decls(types)))
        except Exception as e:  # noqa: BLE001
            record("FAIL", f"Call {i + 1}", f"{type(e).__name__}: {str(e)[:150]}")
            return
        ms = (time.perf_counter() - t) * 1000
        u = resp.usage_metadata
        prompt = getattr(u, "prompt_token_count", 0) or 0
        cached = getattr(u, "cached_content_token_count", 0) or 0
        rows.append((prompt, cached, ms))
        print(f"      call {i + 1}: prompt {prompt:>6} tokens, cached {cached:>6}  ({ms:.0f} ms)")
        time.sleep(2)

    prompt_tokens = rows[0][0]
    record("INFO", "System prompt + tools size", f"~{prompt_tokens} tokens per request")
    hits = [r for r in rows[1:] if r[1] > 0]
    if hits:
        share = statistics.mean(r[1] / r[0] for r in hits)
        record("PASS", "Implicit prompt caching works",
               f"{len(hits)}/{len(rows) - 1} repeat calls hit the cache, ~{share:.0%} of the prompt cached")
    else:
        record("WARN", "Implicit prompt caching not observed",
               "no cached tokens. Common causes: prompt below the model's minimum cacheable size, "
               "or this model/tier doesn't cache implicitly - check Google's caching docs, "
               "consider explicit caching")


# ═════════════════════════════════════════════════════════════════════════════
# D. Final answer with tools switched off
# ═════════════════════════════════════════════════════════════════════════════

def part_d(client, types, model):
    print(f"\nD. 'Limit reached -> answer with what you have' call ({model})")
    system = "You are a VoxelBox technical assistant. Use tools to look things up."
    tools = tool_decls(types)
    question = "Why did patient 2013's DTI job fail?"
    contents = [types.Content(role="user", parts=[types.Part(text=question)])]

    # 1) Get a REAL function call from the model (forced), so any signatures are genuine.
    try:
        r1 = client.models.generate_content(model=model, contents=contents,
                                            config=base_cfg(types, system, tools, mode="ANY"))
    except Exception as e:  # noqa: BLE001
        record("FAIL", "Model produces a tool call", f"{type(e).__name__}: {str(e)[:150]}")
        return
    calls = function_calls(r1)
    if not calls:
        record("FAIL", "Model produces a tool call", "no function_call returned with mode=ANY")
        return
    record("PASS", "Model produces a tool call", calls[0].function_call.name)

    # 2) Add the model turn + a fake tool result, as the agent would.
    contents.append(r1.candidates[0].content)
    contents.append(types.Content(role="user", parts=[types.Part.from_function_response(
        name=calls[0].function_call.name,
        response={"status": "failed", "stage": "registration", "error": "T1 image missing"})]))
    contents.append(types.Content(role="user", parts=[types.Part(
        text="Step limit reached. Answer now using only the information gathered above.")]))

    # 3) Same tools declared, but calling them is switched OFF -> must answer in text.
    t = time.perf_counter()
    try:
        r2 = client.models.generate_content(model=model, contents=contents,
                                            config=base_cfg(types, system, tools, mode="NONE"))
    except Exception as e:  # noqa: BLE001
        record("FAIL", "Final answer with tools off", f"{type(e).__name__}: {str(e)[:200]}")
        return
    ms = (time.perf_counter() - t) * 1000
    text = (r2.text or "").strip()
    if function_calls(r2):
        record("FAIL", "Final answer with tools off", "model still tried to call a tool")
    elif not text:
        record("FAIL", "Final answer with tools off", "empty answer")
    else:
        uses_result = any(w in text.lower() for w in ("registration", "t1"))
        record("PASS" if uses_result else "WARN", "Final answer with tools off",
               f"{ms:.0f} ms, {'uses' if uses_result else 'IGNORES'} the gathered result: {text[:90]!r}")


# ═════════════════════════════════════════════════════════════════════════════
# E. Several tools in one step
# ═════════════════════════════════════════════════════════════════════════════

def part_e(client, types, model):
    print(f"\nE. Several tools in one step ({model})")
    system = ("You are a VoxelBox technical assistant. When a question needs several lookups, "
              "request ALL of them in the same step - do not wait for one result before asking "
              "for the next.")
    question = ("Why did patient 2013's DTI job fail, and what does the DTI acquisition "
                "protocol say the b-value should be?")
    try:
        r = client.models.generate_content(
            model=model, contents=[types.Content(role="user", parts=[types.Part(text=question)])],
            config=base_cfg(types, system, tool_decls(types), mode="AUTO"))
    except Exception as e:  # noqa: BLE001
        record("FAIL", "Parallel tool request", f"{type(e).__name__}: {str(e)[:150]}")
        return
    names = [p.function_call.name for p in function_calls(r)]
    if len(set(names)) >= 2:
        record("PASS", "Parallel tool request", f"both tools requested in one step: {names}")
    elif names:
        record("WARN", "Parallel tool request", f"only {names} in step 1 - needs 2 steps for this question")
    else:
        record("WARN", "Parallel tool request", "model answered without tools")


# ═════════════════════════════════════════════════════════════════════════════

def main():
    try:
        from dotenv import load_dotenv
        load_dotenv(BACKEND / ".env")
    except ImportError:
        pass
    ap = argparse.ArgumentParser(description="Item 5 pretest: latency & cost")
    ap.add_argument("--parts", default="ABCDE", help="which parts to run, e.g. B or CDE")
    ap.add_argument("--api", default="http://localhost:8042")
    ap.add_argument("--hospital", default=None, help="hospital id for data questions (local testing)")
    ap.add_argument("--repeat", type=int, default=1, help="times to send each baseline message")
    ap.add_argument("--compare", help="earlier baseline JSON to compare against")
    ap.add_argument("--model", default=None, help="Gemini model (default: the one in config.yaml)")
    ap.add_argument("--cache-calls", type=int, default=4)
    args = ap.parse_args()
    parts = args.parts.upper()

    model = args.model
    if not model:
        try:
            from config import get_config
            model = get_config().llm.providers["gemini"].model
        except Exception:  # noqa: BLE001
            model = "gemini-2.5-flash"

    print("\nBSAI item 5 pretest (latency & cost)")
    if "B" in parts:
        part_b()
    if "A" in parts:
        part_a(args.api, args.hospital, args.repeat, args.compare)
    if set("CDE") & set(parts):
        client, types = gemini_client()
        if client:
            if "C" in parts:
                part_c(client, types, model, args.cache_calls)
            if "D" in parts:
                part_d(client, types, model)
            if "E" in parts:
                part_e(client, types, model)

    fails = results.count("FAIL")
    print("\n" + "-" * 60)
    print(f"PASS {results.count('PASS')}   FAIL {fails}   WARN {results.count('WARN')}   "
          f"INFO {results.count('INFO')}   SKIP {results.count('SKIP')}")
    print("Ready to implement." if not fails else "Fix the FAILed checks before implementing.")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()