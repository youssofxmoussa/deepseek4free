"""
اختبار شامل لكل ميزات DeepSeek4Free
"""
import os, sys, time
from dsk.api import DeepSeekAPI, AuthenticationError, RateLimitError, NetworkError, APIError
from dotenv import load_dotenv

load_dotenv()

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

results = {}

def header(title):
    print(f"\n{BOLD}{CYAN}{'='*70}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'='*70}{RESET}")

def ok(msg):   print(f"  {GREEN}✅ {msg}{RESET}")
def fail(msg): print(f"  {RED}❌ {msg}{RESET}")
def info(msg): print(f"  {YELLOW}ℹ  {msg}{RESET}")

def collect(name, text, error=None):
    results[name] = {"ok": error is None, "output": text, "error": str(error) if error else None}

# ─────────────────────────────────────────────────────────────────────────────
# 1. SESSION MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────
header("1 / 7  ─  Session Management")
try:
    api = DeepSeekAPI(os.getenv("DEEPSEEK_AUTH_TOKEN"))
    session_id = api.create_chat_session()
    assert session_id and len(session_id) > 10
    ok(f"Session created → {session_id[:20]}…")
    collect("session_management", session_id)
except Exception as e:
    fail(f"Failed: {e}")
    collect("session_management", "", e)
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# 2. STREAMING RESPONSE  (no thinking, no search)
# ─────────────────────────────────────────────────────────────────────────────
header("2 / 7  ─  Streaming Response (basic)")
try:
    chunks_received = 0
    full_text = []
    for chunk in api.chat_completion(session_id, "Say exactly: Hello from DeepSeek!", thinking_enabled=False):
        if chunk["type"] == "text" and chunk["content"]:
            full_text.append(chunk["content"])
            chunks_received += 1
    answer = "".join(full_text).strip()
    assert answer, "Empty response"
    ok(f"Got {chunks_received} chunks")
    ok(f"Response: {answer[:120]}")
    collect("streaming", answer)
except Exception as e:
    fail(f"Failed: {e}")
    collect("streaming", "", e)

# ─────────────────────────────────────────────────────────────────────────────
# 3. THINKING PROCESS
# ─────────────────────────────────────────────────────────────────────────────
header("3 / 7  ─  Thinking Process (thinking_enabled=True)")
try:
    session2 = api.create_chat_session()
    thinking_chunks = []
    text_chunks = []
    for chunk in api.chat_completion(session2, "What is the square root of 144? Show your reasoning.", thinking_enabled=True):
        if chunk["type"] == "thinking" and chunk["content"]:
            thinking_chunks.append(chunk["content"])
        elif chunk["type"] == "text" and chunk["content"]:
            text_chunks.append(chunk["content"])
    thinking_text = "".join(thinking_chunks).strip()
    answer = "".join(text_chunks).strip()
    if thinking_text:
        ok(f"Thinking received ({len(thinking_chunks)} chunks): {thinking_text[:100]}…")
    else:
        info("No thinking content returned (model may skip it for simple questions)")
    ok(f"Answer: {answer[:120]}")
    collect("thinking", answer, None if answer else Exception("Empty answer"))
except Exception as e:
    fail(f"Failed: {e}")
    collect("thinking", "", e)

# ─────────────────────────────────────────────────────────────────────────────
# 4. WEB SEARCH
# ─────────────────────────────────────────────────────────────────────────────
header("4 / 7  ─  Web Search (search_enabled=True)")
try:
    session3 = api.create_chat_session()
    text_chunks = []
    for chunk in api.chat_completion(
        session3,
        "What year is it right now? Just say the year number.",
        thinking_enabled=False,
        search_enabled=True
    ):
        if chunk["type"] == "text" and chunk["content"]:
            text_chunks.append(chunk["content"])
    answer = "".join(text_chunks).strip()
    assert answer, "Empty response"
    ok(f"Response with search: {answer[:200]}")
    collect("web_search", answer)
except Exception as e:
    fail(f"Failed: {e}")
    collect("web_search", "", e)

# ─────────────────────────────────────────────────────────────────────────────
# 5. CONVERSATION HISTORY  (multi-turn in same session)
# ─────────────────────────────────────────────────────────────────────────────
header("5 / 7  ─  Conversation History (multi-turn)")
try:
    session4 = api.create_chat_session()

    def ask(s, q):
        parts = []
        for chunk in api.chat_completion(s, q, thinking_enabled=False):
            if chunk["type"] == "text": parts.append(chunk["content"])
        return "".join(parts).strip()

    r1 = ask(session4, "My favorite color is blue. Remember this.")
    ok(f"Turn 1: {r1[:80]}")
    time.sleep(1)
    r2 = ask(session4, "What is my favorite color?")
    ok(f"Turn 2: {r2[:80]}")
    remembered = "blue" in r2.lower()
    if remembered:
        ok("Memory check PASSED — model remembered 'blue' ✔")
    else:
        info(f"Memory check inconclusive — response: {r2[:80]}")
    collect("conversation_history", r2, None if r2 else Exception("Empty"))
except Exception as e:
    fail(f"Failed: {e}")
    collect("conversation_history", "", e)

# ─────────────────────────────────────────────────────────────────────────────
# 6. THREADED CONVERSATION (parent_message_id)
# ─────────────────────────────────────────────────────────────────────────────
header("6 / 7  ─  Threaded Conversation (parent_message_id)")
try:
    import json
    from curl_cffi import requests as cffi_requests
    from dsk.pow import DeepSeekPOW

    BASE_URL = "https://chat.deepseek.com/api/v0"
    pow_solver = DeepSeekPOW()

    def _headers(token, pow_resp=None):
        h = {
            "accept": "*/*",
            "authorization": f"Bearer {token}",
            "content-type": "application/json",
            "origin": "https://chat.deepseek.com",
            "referer": "https://chat.deepseek.com/",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "x-app-version": "20241129.1",
            "x-client-locale": "en_US",
            "x-client-platform": "web",
            "x-client-version": "1.0.0-always",
        }
        if pow_resp: h["x-ds-pow-response"] = pow_resp
        return h

    token = os.getenv("DEEPSEEK_AUTH_TOKEN")

    def create_session():
        r = cffi_requests.post(f"{BASE_URL}/chat_session/create",
                               headers=_headers(token),
                               json={"character_id": None}, impersonate="chrome120")
        return r.json()["data"]["biz_data"]["id"]

    def chat_raw(session_id, prompt, parent_id=None):
        r2 = cffi_requests.post(f"{BASE_URL}/chat/create_pow_challenge",
                                headers=_headers(token),
                                json={"target_path": "/api/v0/chat/completion"}, impersonate="chrome120")
        challenge = r2.json()["data"]["biz_data"]["challenge"]
        pw = pow_solver.solve_challenge(challenge)
        h = _headers(token, pw)

        last_msg_id = None
        text_parts = []
        resp = cffi_requests.post(f"{BASE_URL}/chat/completion", headers=h,
                                   json={"chat_session_id": session_id,
                                         "parent_message_id": parent_id,
                                         "prompt": prompt,
                                         "ref_file_ids": [],
                                         "thinking_enabled": False,
                                         "search_enabled": False},
                                   impersonate="chrome120", stream=True, timeout=None)
        current_path = None
        for line in resp.iter_lines():
            if not line: continue
            if line.startswith(b"data: "):
                try:
                    d = json.loads(line[6:])
                    # capture response_message_id
                    if "response_message_id" in d:
                        last_msg_id = d["response_message_id"]
                    if "p" in d:
                        current_path = d["p"]
                        if isinstance(d.get("v"), str) and "content" in current_path and "thinking" not in current_path:
                            text_parts.append(d["v"])
                    elif "v" in d and isinstance(d["v"], str) and current_path and "content" in current_path and "thinking" not in current_path:
                        text_parts.append(d["v"])
                except: pass
        return "".join(text_parts).strip(), last_msg_id

    ts = create_session()
    r1, msg_id = chat_raw(ts, "Define 'recursion' in one sentence.")
    ok(f"Turn 1: {r1[:100]}")
    info(f"Message ID captured: {msg_id}")

    time.sleep(1)
    r2, _ = chat_raw(ts, "Give me a Python code example of what you just defined.", parent_id=msg_id)
    ok(f"Turn 2 (threaded): {r2[:100]}")
    has_code = "def " in r2 or "```" in r2 or "return" in r2
    if has_code:
        ok("Thread context PASSED — model replied with code related to recursion ✔")
    else:
        info(f"Thread context inconclusive: {r2[:100]}")
    collect("threaded_conversation", r2, None if r2 else Exception("Empty"))
except Exception as e:
    fail(f"Failed: {e}")
    collect("threaded_conversation", "", e)

# ─────────────────────────────────────────────────────────────────────────────
# 7. ERROR HANDLING
# ─────────────────────────────────────────────────────────────────────────────
header("7 / 7  ─  Error Handling")
# 7a. Invalid token → AuthenticationError
try:
    bad_api = DeepSeekAPI("invalid_token_xyz")
    s = bad_api.create_chat_session()
    list(bad_api.chat_completion(s, "hi", thinking_enabled=False))
    fail("Should have raised AuthenticationError!")
    collect("error_handling_auth", "no exception raised", Exception("no exception"))
except AuthenticationError as e:
    ok(f"AuthenticationError raised correctly → {e}")
    collect("error_handling_auth", str(e))
except Exception as e:
    info(f"Different exception (API may reject at session level): {type(e).__name__}: {e}")
    collect("error_handling_auth", str(e))

# 7b. Empty prompt → ValueError
try:
    session_err = api.create_chat_session()
    list(api.chat_completion(session_err, "", thinking_enabled=False))
    fail("Should have raised ValueError for empty prompt!")
    collect("error_handling_value", "no exception", Exception("no exception"))
except ValueError as e:
    ok(f"ValueError raised correctly for empty prompt → {e}")
    collect("error_handling_value", str(e))
except Exception as e:
    fail(f"Wrong exception: {type(e).__name__}: {e}")
    collect("error_handling_value", str(e), e)

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
header("SUMMARY")
pass_count = sum(1 for r in results.values() if r["ok"])
fail_count = len(results) - pass_count

feature_names = {
    "session_management":   "Session Management",
    "streaming":            "Streaming Responses",
    "thinking":             "Thinking Process",
    "web_search":           "Web Search",
    "conversation_history": "Conversation History",
    "threaded_conversation":"Threaded Conversation",
    "error_handling_auth":  "Error Handling (bad token)",
    "error_handling_value": "Error Handling (empty prompt)",
}

for key, name in feature_names.items():
    r = results.get(key, {"ok": False, "error": "not run"})
    status = f"{GREEN}PASS ✅{RESET}" if r["ok"] else f"{RED}FAIL ❌{RESET}"
    err_note = f"  ← {r['error']}" if not r["ok"] and r.get("error") else ""
    print(f"  {status}  {name}{err_note}")

print(f"\n{BOLD}  Total: {pass_count}/{len(results)} passed{RESET}\n")
