"""
Context Window Tester
Tests how well the chatbot remembers conversation history
Run: python test_context.py
"""

import requests
import time
import json

API_URL = "http://localhost:8000/api/chat/"
MEMORY_URL = "http://localhost:8000/api/chat/memory/info/"

# Use a unique session for each test run
SESSION_ID = f"context_test_{int(time.time())}"

def send_message(message, delay=1):
    """Send a message and return the response"""
    try:
        response = requests.post(API_URL, json={
            "message": message,
            "session_id": SESSION_ID,
            "user_id": "tester"
        }, timeout=30)

        if response.status_code == 200:
            data = response.json()
            time.sleep(delay)
            return data.get("response", "")
        else:
            print(f"  ❌ HTTP Error: {response.status_code}")
            return None
    except Exception as e:
        print(f"  ❌ Request failed: {e}")
        return None

def get_memory_state():
    """Get current memory state from backend"""
    try:
        response = requests.get(f"{MEMORY_URL}{SESSION_ID}", timeout=10)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def print_result(test_name, passed, detail=""):
    """Print test result"""
    icon = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {icon} - {test_name}")
    if detail:
        print(f"         → {detail}")

def print_header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def print_memory_state():
    """Print current memory state"""
    state = get_memory_state()
    if state and "current_state" in state:
        s = state["current_state"]
        cfg = state["config"]
        print(f"\n  📊 Memory State:")
        print(f"     Messages in memory : {s['message_count']}")
        print(f"     Estimated tokens   : {s['estimated_tokens']}")
        print(f"     Token usage        : {s['token_percentage']:.1f}%")
        print(f"     Max token limit    : {cfg['max_tokens']}")
        print(f"     Window size (k)    : {cfg['buffer_window_k']}")
    else:
        print("  ⚠️  Could not fetch memory state")


# ─────────────────────────────────────────────
#  TEST 1: Basic Memory
# ─────────────────────────────────────────────
def test_basic_memory():
    print_header("TEST 1: Basic Memory (Can it remember a fact?)")

    print("  → Sending: 'My lucky number is 7777'")
    send_message("My lucky number is 7777")

    print("  → Asking:  'What is my lucky number?'")
    reply = send_message("What is my lucky number?")

    passed = reply and "7777" in reply
    print_result("Remembered lucky number 7777", passed, reply[:100] if reply else "No response")
    return passed


# ─────────────────────────────────────────────
#  TEST 2: Multi-turn Context Chain
# ─────────────────────────────────────────────
def test_context_chain():
    print_header("TEST 2: Context Chain (Multi-turn understanding)")

    print("  → Message 1: 'I am working on a project called APOLLO'")
    send_message("I am working on a project called APOLLO")

    print("  → Message 2: 'It is about brain imaging research'")
    send_message("It is about brain imaging research")

    print("  → Asking: 'What is my project about?'")
    reply = send_message("What is my project about?")

    passed = reply and ("APOLLO" in reply or "brain" in reply.lower() or "imaging" in reply.lower())
    print_result("Remembered project name and topic", passed, reply[:120] if reply else "No response")
    return passed


# ─────────────────────────────────────────────
#  TEST 3: Memory Limit (k window)
# ─────────────────────────────────────────────
def test_memory_limit():
    print_header("TEST 3: Memory Limit (Does it forget after k exchanges?)")

    # Get current k value
    state = get_memory_state()
    k = state["config"]["buffer_window_k"] if state else 5
    print(f"  → Current k value: {k} (will test with {k+2} messages)")

    print(f"  → Planting secret: 'SECRET_CODE = ALPHA999'")
    send_message("Remember this secret code: ALPHA999")

    # Push k+1 more messages to overflow the window
    print(f"  → Sending {k+1} filler messages to push secret out of window...")
    fillers = [
        "What is structural MRI?",
        "How does fMRI work?",
        "What is diffusion MRI?",
        "Explain tractography",
        "What is connectomics?",
        "What are brain parcellations?",
        "Explain white matter tracts",
        "What is resting state fMRI?",
        "What is BOLD signal?",
        "Explain grey matter segmentation"
    ]

    for i, msg in enumerate(fillers[:k+1]):
        print(f"     Filler {i+1}/{k+1}: '{msg[:40]}'")
        send_message(msg, delay=0.5)

    print("  → Now asking: 'What was the secret code I gave you?'")
    reply = send_message("What was the secret code I gave you?")

    forgot = reply and "ALPHA999" not in reply
    print_result(
        f"Correctly forgot secret after {k+1} exchanges (k={k})",
        forgot,
        f"Reply: {reply[:100]}" if reply else "No response"
    )

    if not forgot and reply:
        print("  ⚠️  Still remembered! Context window might be larger than expected")

    return forgot


# ─────────────────────────────────────────────
#  TEST 4: Token Usage Check
# ─────────────────────────────────────────────
def test_token_usage():
    print_header("TEST 4: Token Usage (Is token tracking working?)")

    print("  → Sending a long message to increase token usage...")
    long_msg = (
        "Please explain in detail the full neuroimaging pipeline including "
        "structural MRI preprocessing, functional MRI analysis, diffusion MRI "
        "tractography, connectome construction, and quality control steps. "
        "Include information about software tools used in each step."
    )
    send_message(long_msg)

    state = get_memory_state()
    if state and "current_state" in state:
        tokens = state["current_state"]["estimated_tokens"]
        pct = state["current_state"]["token_percentage"]
        max_t = state["config"]["max_tokens"]

        passed = tokens > 0
        print_result("Token counting is active", passed,
                     f"{tokens} tokens used ({pct:.1f}% of {max_t} max)")
    else:
        print_result("Token counting is active", False, "Could not read memory state")
        passed = False

    return passed


# ─────────────────────────────────────────────
#  TEST 5: Session Isolation
# ─────────────────────────────────────────────
def test_session_isolation():
    print_header("TEST 5: Session Isolation (Different sessions = different memory)")

    session_a = f"session_A_{int(time.time())}"
    session_b = f"session_B_{int(time.time())}"

    print("  → Session A: 'My name is Alice'")
    requests.post(API_URL, json={"message": "My name is Alice", "session_id": session_a}, timeout=30)

    print("  → Session B: 'My name is Bob'")
    requests.post(API_URL, json={"message": "My name is Bob", "session_id": session_b}, timeout=30)

    print("  → Session A asking: 'What is my name?'")
    r_a = requests.post(API_URL, json={"message": "What is my name?", "session_id": session_a}, timeout=30)

    print("  → Session B asking: 'What is my name?'")
    r_b = requests.post(API_URL, json={"message": "What is my name?", "session_id": session_b}, timeout=30)

    reply_a = r_a.json().get("response", "") if r_a.status_code == 200 else ""
    reply_b = r_b.json().get("response", "") if r_b.status_code == 200 else ""

    a_correct = "Alice" in reply_a and "Bob" not in reply_a
    b_correct = "Bob" in reply_b and "Alice" not in reply_b

    print_result("Session A knows Alice (not Bob)", a_correct, reply_a[:80])
    print_result("Session B knows Bob (not Alice)", b_correct, reply_b[:80])

    return a_correct and b_correct


# ─────────────────────────────────────────────
#  MAIN RUNNER
# ─────────────────────────────────────────────
def main():
    print("\n" + "🧪 " * 20)
    print("     CONTEXT WINDOW TESTER")
    print("🧪 " * 20)
    print(f"\n  Session ID : {SESSION_ID}")
    print(f"  Backend    : {API_URL}")

    # Check backend is running
    try:
        r = requests.get("http://localhost:8000/health", timeout=5)
        if r.status_code != 200:
            print("\n❌ Backend is not running! Start it with: python main.py")
            return
        print("  Backend    : ✅ Online\n")
    except:
        print("\n❌ Cannot reach backend at localhost:8000")
        print("   Start it with: cd backend && python main.py")
        return

    results = {}

    results["basic_memory"]       = test_basic_memory()
    print_memory_state()

    results["context_chain"]      = test_context_chain()
    print_memory_state()

    results["memory_limit"]       = test_memory_limit()
    print_memory_state()

    results["token_usage"]        = test_token_usage()
    print_memory_state()

    results["session_isolation"]  = test_session_isolation()

    # ── Summary ──────────────────────────────
    print_header("FINAL SUMMARY")
    total   = len(results)
    passed  = sum(1 for v in results.values() if v)

    for test, result in results.items():
        icon = "✅" if result else "❌"
        print(f"  {icon}  {test.replace('_', ' ').title()}")

    print(f"\n  Score: {passed}/{total} tests passed")

    if passed == total:
        print("\n  🎉 Context window is working PERFECTLY!")
    elif passed >= total * 0.7:
        print("\n  ⚠️  Context window is mostly working, check failed tests.")
    else:
        print("\n  ❌ Context window has issues. Check config and restart backend.")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()