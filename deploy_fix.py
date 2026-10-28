#!/usr/bin/env python3
"""
BrainSightAI — All-in-one Diagnose + Fix Script
================================================
Run from your project root on the cloud server:

    python3 deploy_fix.py

It will:
  1. Diagnose all known issues
  2. Auto-fix everything it can
  3. Restart the backend
  4. Confirm the fixes worked
"""

import os
import sys
import re
import time
import shutil
import socket
import subprocess
import urllib.request
import urllib.error
import json

# ── CONFIG ────────────────────────────────────────────────────
BACKEND_PORT     = 8042
FRONTEND_PORT    = 8043
BACKEND_HOST     = f"http://localhost:{BACKEND_PORT}"
FRONTEND_ORIGIN  = f"http://4.213.16.34:{FRONTEND_PORT}"
REQUIRED_PKGS    = ["fastapi", "uvicorn", "slowapi", "google-genai",
                    "langchain", "pymongo", "chromadb", "pyyaml"]

# ── COLORS ────────────────────────────────────────────────────
G  = "\033[92m"   # green
R  = "\033[91m"   # red
Y  = "\033[93m"   # yellow
C  = "\033[96m"   # cyan
B  = "\033[1m"    # bold
NC = "\033[0m"    # reset

def ok(msg):   print(f"  {G}[PASS]{NC}  {msg}")
def fail(msg): print(f"  {R}[FAIL]{NC}  {msg}")
def fix(msg):  print(f"  {G}[FIX]{NC}   {msg}")
def warn(msg): print(f"  {Y}[WARN]{NC}  {msg}")
def info(msg): print(f"  {C}[INFO]{NC}  {msg}")
def head(msg): print(f"\n{B}{C}{'='*60}{NC}\n{B}{C}  {msg}{NC}\n{B}{C}{'='*60}{NC}")
def section(n, msg): print(f"\n{B}[{n}] {msg}{NC}")

# ── HELPERS ───────────────────────────────────────────────────

def find_file(*candidates):
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None

def is_port_open(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(2)
        return s.connect_ex(("127.0.0.1", port)) == 0

def http_get(url, timeout=5):
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()
    except Exception as e:
        return None, str(e)

def http_options(url, origin, timeout=5):
    """Send a CORS preflight request."""
    try:
        req = urllib.request.Request(url, method="OPTIONS")
        req.add_header("Origin", origin)
        req.add_header("Access-Control-Request-Method", "POST")
        req.add_header("Access-Control-Request-Headers", "Content-Type")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers)
    except Exception as e:
        return None, {}

def http_post(url, body, origin, timeout=30):
    try:
        data = json.dumps(body).encode()
        req  = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Origin", origin)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()
    except Exception as e:
        return None, str(e)

def pkg_installed(name):
    result = subprocess.run(
        [sys.executable, "-m", "pip", "show", name],
        capture_output=True, text=True
    )
    return result.returncode == 0

def pkg_version(name):
    result = subprocess.run(
        [sys.executable, "-m", "pip", "show", name],
        capture_output=True, text=True
    )
    for line in result.stdout.splitlines():
        if line.startswith("Version:"):
            return line.split(":", 1)[1].strip()
    return "unknown"

# ── ISSUE TRACKER ─────────────────────────────────────────────
issues   = []   # list of (label, fix_fn)
findings = []   # diagnostic results

def add_issue(label, fix_fn):
    issues.append((label, fix_fn))

# ══════════════════════════════════════════════════════════════
#  PHASE 1 — DIAGNOSE
# ══════════════════════════════════════════════════════════════

def diagnose():
    head("PHASE 1 — DIAGNOSING")

    # ── 1. Ports ─────────────────────────────────────────────
    section(1, "Checking ports")

    if is_port_open(BACKEND_PORT):
        ok(f"Backend is listening on :{BACKEND_PORT}")
    else:
        fail(f"Backend NOT listening on :{BACKEND_PORT}")
        add_issue(
            "Backend port not open — will restart uvicorn",
            fix_restart_backend
        )

    if is_port_open(FRONTEND_PORT):
        ok(f"Frontend is listening on :{FRONTEND_PORT}")
    else:
        warn(f"Frontend NOT listening on :{FRONTEND_PORT} — start npm later")

    # ── 2. Health endpoint ───────────────────────────────────
    section(2, "Testing /health endpoint")

    status, body = http_get(f"{BACKEND_HOST}/health")
    if status == 200:
        ok(f"GET /health → {status}")
        info(f"Body: {body[:200]}")
    elif status is None:
        fail(f"Could not reach backend: {body}")
    else:
        fail(f"GET /health → {status}")
        info(f"Body: {body[:200]}")

    # ── 3. CORS preflight ────────────────────────────────────
    section(3, "Testing CORS preflight (OPTIONS /api/chat/)")

    cors_status, cors_headers = http_options(
        f"{BACKEND_HOST}/api/chat/", FRONTEND_ORIGIN
    )

    if cors_status in (200, 204):
        ok(f"CORS preflight → {cors_status}")
    elif cors_status in (400, 422, 429):
        fail(f"CORS preflight → {cors_status}  ← this blocks the frontend!")
        add_issue(
            "Rate limiter returning 400 on OPTIONS preflight",
            fix_rate_limiter
        )
    else:
        fail(f"CORS preflight → {cors_status}")
        add_issue(
            "CORS preflight failing — will patch main.py",
            fix_rate_limiter
        )

    acao = cors_headers.get("Access-Control-Allow-Origin", "")
    if acao:
        ok(f"Access-Control-Allow-Origin: {acao}")
    else:
        fail("No Access-Control-Allow-Origin header in response")

    # ── 4. Chat POST ─────────────────────────────────────────
    section(4, "Testing POST /api/chat/")

    chat_status, chat_body = http_post(
        f"{BACKEND_HOST}/api/chat/",
        {"message": "hello", "session_id": "diag-001"},
        FRONTEND_ORIGIN
    )

    if chat_status == 200:
        ok(f"POST /api/chat/ → {chat_status}")
        info(f"Response: {chat_body[:200]}")
    else:
        fail(f"POST /api/chat/ → {chat_status}")
        info(f"Body: {chat_body[:300]}")

    # ── 5. config.yaml ───────────────────────────────────────
    section(5, "Checking config.yaml allow_origins")

    config_path = find_file(
        "./backend/config.yaml", "./config.yaml", "../backend/config.yaml"
    )

    if config_path:
        content = open(config_path).read()
        if FRONTEND_ORIGIN in content:
            ok(f"{FRONTEND_ORIGIN} found in config.yaml")
        else:
            fail(f"{FRONTEND_ORIGIN} NOT in config.yaml allow_origins")
            add_issue(
                "Frontend origin missing from config.yaml",
                fix_config_yaml
            )
    else:
        warn("config.yaml not found — skipping")

    # ── 6. main.py rate limiter ──────────────────────────────
    section(6, "Checking main.py rate limiter")

    main_path = find_file(
        "./backend/main.py", "./main.py", "../backend/main.py"
    )

    if main_path:
        content = open(main_path).read()
        if "preflight-exempt" in content or \
           ("OPTIONS" in content and "_rate_limit_key" in content):
            ok("Rate limiter already exempts OPTIONS requests")
        else:
            fail("Rate limiter does NOT exempt OPTIONS — causes 400 on preflight")
            add_issue(
                "main.py rate limiter must skip OPTIONS requests",
                fix_rate_limiter
            )
    else:
        warn("main.py not found — skipping")

    # ── 7. Python packages ───────────────────────────────────
    section(7, "Checking Python packages")

    missing_pkgs = []
    for pkg in REQUIRED_PKGS:
        if pkg_installed(pkg):
            ok(f"{pkg} == {pkg_version(pkg)}")
        else:
            fail(f"{pkg} NOT installed")
            missing_pkgs.append(pkg)

    if missing_pkgs:
        add_issue(
            f"Missing packages: {', '.join(missing_pkgs)}",
            lambda pkgs=missing_pkgs: fix_packages(pkgs)
        )

    return len(issues)


# ══════════════════════════════════════════════════════════════
#  PHASE 2 — FIX FUNCTIONS
# ══════════════════════════════════════════════════════════════

def fix_rate_limiter():
    """Patch main.py to exempt OPTIONS from rate limiting."""
    main_path = find_file(
        "./backend/main.py", "./main.py", "../backend/main.py"
    )
    if not main_path:
        fail("Cannot find main.py — skipping rate limiter fix")
        return False

    # Backup
    backup = main_path + ".bak"
    shutil.copy2(main_path, backup)
    info(f"Backed up to {backup}")

    with open(main_path, "r") as f:
        content = f.read()

    OLD_SIMPLE  = "# Initialize rate limiter\nlimiter = Limiter(key_func=get_remote_address)"
    OLD_VARIANT = "limiter = Limiter(key_func=get_remote_address)"
    NEW_CODE = """\
# Initialize rate limiter — exempt CORS preflight (OPTIONS) from rate limiting
def _rate_limit_key(request):
    if request.method == "OPTIONS":
        return "preflight-exempt"
    from slowapi.util import get_remote_address as _gra
    return _gra(request)

limiter = Limiter(key_func=_rate_limit_key)"""

    if OLD_SIMPLE in content:
        content = content.replace(OLD_SIMPLE, NEW_CODE)
        fix("Replaced limiter block (exact match)")
    elif OLD_VARIANT in content and "_rate_limit_key" not in content:
        content = content.replace(OLD_VARIANT, NEW_CODE)
        fix("Replaced limiter line (variant match)")
    else:
        # regex fallback
        pattern = r'(#\s*Initialize rate limiter.*?\n)?(limiter\s*=\s*Limiter\(key_func=get_remote_address\))'
        new_content, n = re.subn(pattern, NEW_CODE, content)
        if n:
            content = new_content
            fix("Replaced via regex fallback")
        else:
            fail("Could not auto-patch main.py — patch manually:")
            print("""
  Find:
    limiter = Limiter(key_func=get_remote_address)

  Replace with:
    def _rate_limit_key(request):
        if request.method == "OPTIONS":
            return "preflight-exempt"
        from slowapi.util import get_remote_address as _gra
        return _gra(request)

    limiter = Limiter(key_func=_rate_limit_key)
""")
            return False

    with open(main_path, "w") as f:
        f.write(content)

    fix(f"main.py patched at {main_path}")
    return True


def fix_config_yaml():
    """Ensure frontend origin is in config.yaml allow_origins."""
    config_path = find_file(
        "./backend/config.yaml", "./config.yaml", "../backend/config.yaml"
    )
    if not config_path:
        fail("Cannot find config.yaml")
        return False

    with open(config_path, "r") as f:
        lines = f.readlines()

    if any(FRONTEND_ORIGIN in line for line in lines):
        info("Origin already present in config.yaml")
        return True

    new_lines = []
    inserted  = False
    for line in lines:
        new_lines.append(line)
        if "allow_origins:" in line and not inserted:
            indent = "        "
            new_lines.append(f'{indent}- "{FRONTEND_ORIGIN}"  # auto-added by deploy_fix.py\n')
            inserted = True

    if inserted:
        with open(config_path, "w") as f:
            f.writelines(new_lines)
        fix(f"Added {FRONTEND_ORIGIN} to {config_path}")
        return True
    else:
        fail("Could not locate allow_origins section in config.yaml")
        return False


def fix_packages(missing):
    """Install missing Python packages."""
    for pkg in missing:
        info(f"Installing {pkg}...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", pkg, "-q"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            fix(f"{pkg} installed")
        else:
            fail(f"Failed to install {pkg}: {result.stderr[:200]}")


def fix_restart_backend():
    """Kill existing uvicorn and restart on BACKEND_PORT."""
    info("Stopping existing backend process if any...")

    # Find and kill process using the port
    result = subprocess.run(
        ["lsof", "-ti", f":{BACKEND_PORT}"],
        capture_output=True, text=True
    )
    pids = result.stdout.strip().split()
    for pid in pids:
        try:
            os.kill(int(pid), 15)
            info(f"Killed PID {pid}")
        except Exception:
            pass

    time.sleep(1)

    # Find backend dir
    backend_dir = "./backend" if os.path.isdir("./backend") else "."

    log_path = "/tmp/bsai_backend.log"
    cmd = f"cd {backend_dir} && uvicorn main:app --host 0.0.0.0 --port {BACKEND_PORT}"
    info(f"Starting: {cmd}")
    info(f"Logs: {log_path}")

    with open(log_path, "w") as log:
        subprocess.Popen(
            cmd, shell=True,
            stdout=log, stderr=log,
            preexec_fn=os.setsid if hasattr(os, 'setsid') else None
        )

    # Wait up to 10 seconds
    for i in range(10):
        time.sleep(1)
        if is_port_open(BACKEND_PORT):
            fix(f"Backend is up on :{BACKEND_PORT}")
            return True
        info(f"Waiting... ({i+1}s)")

    fail(f"Backend did not start in 10s — check logs: tail -f {log_path}")
    return False


# ══════════════════════════════════════════════════════════════
#  PHASE 3 — VERIFY
# ══════════════════════════════════════════════════════════════

def verify():
    head("PHASE 3 — VERIFYING FIXES")
    passed = 0
    total  = 4

    section("A", f"Port :{BACKEND_PORT}")
    if is_port_open(BACKEND_PORT):
        ok(f"Backend listening on :{BACKEND_PORT}"); passed += 1
    else:
        fail(f"Backend still not listening on :{BACKEND_PORT}")

    section("B", "Health endpoint")
    status, body = http_get(f"{BACKEND_HOST}/health")
    if status == 200:
        ok(f"GET /health → {status}"); passed += 1
    else:
        fail(f"GET /health → {status}")

    section("C", "CORS preflight")
    cors_status, cors_headers = http_options(
        f"{BACKEND_HOST}/api/chat/", FRONTEND_ORIGIN
    )
    if cors_status in (200, 204):
        ok(f"OPTIONS /api/chat/ → {cors_status}"); passed += 1
        acao = cors_headers.get("Access-Control-Allow-Origin", "")
        if acao:
            ok(f"Access-Control-Allow-Origin: {acao}")
    else:
        fail(f"OPTIONS /api/chat/ → {cors_status} — CORS still broken")

    section("D", "Chat POST")
    chat_status, chat_body = http_post(
        f"{BACKEND_HOST}/api/chat/",
        {"message": "hello", "session_id": "verify-001"},
        FRONTEND_ORIGIN
    )
    if chat_status == 200:
        ok(f"POST /api/chat/ → {chat_status}"); passed += 1
    else:
        fail(f"POST /api/chat/ → {chat_status}")
        info(f"Body: {chat_body[:300]}")

    print()
    score_color = G if passed == total else Y if passed >= 2 else R
    print(f"  {score_color}{B}Score: {passed}/{total} checks passing{NC}")

    if passed == total:
        print(f"\n  {G}{B}All checks passed! Open http://4.213.16.34:{FRONTEND_PORT} in your browser.{NC}\n")
    else:
        print(f"\n  {Y}Some issues remain — check logs above.{NC}")
        print(f"  Backend logs: tail -f /tmp/bsai_backend.log\n")


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    head("BrainSightAI — All-in-one Deploy Fix")
    print(f"  Backend : {BACKEND_HOST}")
    print(f"  Frontend: {FRONTEND_ORIGIN}")
    print(f"  Python  : {sys.version.split()[0]}")

    # Phase 1: Diagnose
    n_issues = diagnose()

    if n_issues == 0:
        head("All checks passed — nothing to fix!")
        verify()
        return

    # Phase 2: Fix
    head(f"PHASE 2 — FIXING {n_issues} ISSUE(S)")
    for label, fix_fn in issues:
        section("→", label)
        fix_fn()

    # Always restart backend after fixes
    section("→", "Restarting backend to apply patches")
    fix_restart_backend()

    # Phase 3: Verify
    verify()


if __name__ == "__main__":
    main()
