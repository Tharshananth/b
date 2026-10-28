"""
LLM provider API check - is each configured provider actually answering?

Run from anywhere (the script anchors itself to backend/):

    python scripts/check_llm_apis.py                     # every enabled provider
    python scripts/check_llm_apis.py --provider gemini   # just one
    python scripts/check_llm_apis.py --timeout 30        # per-provider limit (s)

For each provider in config.yaml it checks:
  1. KEY    - the api_key_env variable is set (value is masked)
  2. CALL   - a one-line prompt through the app's own provider class succeeds
  3. TIME   - how long the round trip took

Failures are classified (bad key / no credit / overloaded / bad model / timeout)
so the fix is obvious. Costs a handful of tokens per provider; no data is sent.
"""
import argparse
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from pathlib import Path

# The app loads config.yaml by relative path, so anchor to backend/ regardless
# of where this was launched from.
BACKEND = Path(__file__).resolve().parent.parent
os.chdir(BACKEND)
sys.path.insert(0, str(BACKEND))

from dotenv import load_dotenv
load_dotenv(BACKEND / ".env")

OK, FAIL, WARN, INFO = "[OK]  ", "[FAIL]", "[WARN]", "      "

PROMPT = "Reply with exactly the word: OK"

# (substring in the error text, short reason, what to do). First match wins.
DIAGNOSES = [
    ("invalid x-api-key", "invalid API key", "replace the key in backend/.env"),
    ("api key not valid", "invalid API key", "replace the key in backend/.env"),
    ("invalid_api_key", "invalid API key", "replace the key in backend/.env"),
    ("401", "authentication failed", "check the key in backend/.env"),
    ("credit", "no credit left", "add credit / billing on the provider account"),
    ("insufficient_quota", "quota exhausted", "add credit / billing on the provider account"),
    ("resource_exhausted", "quota / rate limit", "wait, or raise the quota"),
    ("429", "rate limited", "wait a minute and retry"),
    ("503", "provider overloaded", "temporary - retry in a few minutes"),
    ("overloaded", "provider overloaded", "temporary - retry in a few minutes"),
    ("unavailable", "provider unavailable", "temporary - retry in a few minutes"),
    ("not_found", "model not found", "fix the model name in config.yaml"),
    ("404", "model not found", "fix the model name in config.yaml"),
    ("permission", "no access to this model", "check the account's model access"),
    ("timed out", "timed out", "network issue or provider very slow"),
]


def diagnose(error):
    low = (error or "").lower()
    for needle, reason, fix in DIAGNOSES:
        if needle in low:
            return reason, fix
    return "unknown error", "see the raw error below"


def mask(value):
    if not value:
        return "(not set)"
    return value[:4] + "..." + value[-4:] if len(value) > 12 else "***"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--provider", help="test only this provider")
    p.add_argument("--timeout", type=int, default=60, help="seconds per provider")
    return p.parse_args()


args = parse_args()

import config
from llm.base import Message
from llm.factory import LLMFactory

llm_cfg = config.get_config().llm
names = [args.provider] if args.provider else list(llm_cfg.providers)

print("-" * 68)
print("LLM PROVIDER CHECK")
print("-" * 68)
print(INFO + f" default provider : {llm_cfg.default_provider}")
print(INFO + f" fallback order   : {', '.join(llm_cfg.fallback_order)}")

factory = LLMFactory()
# Providers log full tracebacks on failure; this script reports them itself.
logging.disable(logging.CRITICAL)
results = {}

for name in names:
    pcfg = llm_cfg.providers.get(name)
    print(f"\n== {name}")
    if pcfg is None:
        print(FAIL + f" '{name}' is not in config.yaml")
        results[name] = False
        continue
    print(INFO + f" model : {pcfg.model}")

    if not pcfg.enabled:
        print(WARN + " disabled in config.yaml - skipped")
        continue

    key = os.getenv(pcfg.api_key_env)
    if not key:
        print(FAIL + f" {pcfg.api_key_env} is not set in backend/.env")
        results[name] = False
        continue
    print(OK + f" {pcfg.api_key_env} = {mask(key)}")

    provider = factory.get_provider(name)
    if provider is None:
        print(FAIL + " provider failed to initialise (see log above)")
        results[name] = False
        continue

    # Run in a thread so one hung provider cannot hang the whole check.
    start = time.time()
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(provider.generate_response,
                         [Message(role="user", content=PROMPT)])
    try:
        resp = future.result(timeout=args.timeout)
        error = resp.error if resp.finish_reason == "error" else None
    except FutureTimeout:
        resp, error = None, f"timed out after {args.timeout}s"
    except Exception as e:
        resp, error = None, f"{type(e).__name__}: {e}"
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    took = time.time() - start

    if error:
        reason, fix = diagnose(error)
        print(FAIL + f" {reason}  ({took:.1f}s)")
        print(INFO + f" fix   : {fix}")
        print(INFO + f" raw   : {' '.join(error.split())[:300]}")
        results[name] = False
    else:
        reply = (resp.content or "").strip().replace("\n", " ")[:60]
        print(OK + f" answered in {took:.1f}s  ->  {reply!r}")
        if took > 20:
            print(WARN + " slow - chat requests will feel sluggish")
        results[name] = True

# -- Summary ----------------------------------------------------------------
print("\n" + "-" * 68)
print("SUMMARY")
print("-" * 68)
for name, ok in results.items():
    print((OK if ok else FAIL) + f" {name}")

working = [n for n, ok in results.items() if ok]
if not working:
    print(FAIL + " no provider is working - the chatbot cannot answer anything")
    sys.exit(1)

chain = [llm_cfg.default_provider] + [p for p in llm_cfg.fallback_order
                                      if p != llm_cfg.default_provider]
first = next((p for p in chain if results.get(p)), None)
if first:
    print(INFO + f" chat will be answered by: {first}")
if not all(results.values()):
    sys.exit(1)
