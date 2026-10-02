#!/usr/bin/env bash
# kit/bench/smoke-test.sh [BASE_URL] — quick correctness check of a running server: arithmetic, a fact, code,
# thinking mode and a tool call. Exits non-zero if any answer is wrong. Takes about a minute.
# A recipe's scripts/smoke-test.sh runs this; a recipe whose model needs different checks replaces that wrapper.
# SMOKE_NO_THINK=1 sends `enable_thinking: false` with the three direct-answer checks (arithmetic, fact, code), for
# models that think past their 256-400 token budgets with thinking on (Qwen3.6-35B-A3B); the thinking check is unchanged.
set -uo pipefail
B="${1:-http://127.0.0.1:${PORT:-8000}/v1}"
python3 - "$B" <<'PY'
import json, os, re, sys, urllib.request
B = sys.argv[1]
DIRECT = {"chat_template_kwargs": {"enable_thinking": False}} if os.environ.get("SMOKE_NO_THINK") == "1" else {}
model = json.load(urllib.request.urlopen(B + "/models", timeout=10))["data"][0]["id"]
def chat(msg, **kw):
    body = {"model": model, "messages": [{"role": "user", "content": msg}], "max_tokens": kw.pop("max_tokens", 256),
            "temperature": 0, **kw}
    req = urllib.request.Request(B + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["message"]
ok = True
def check(name, cond, got):
    global ok
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + ("" if cond else f"   got: {str(got)[:200]!r}")); ok &= bool(cond)
m = chat("What is 17 * 23? Reply with the number only.", **DIRECT); check("arithmetic", "391" in (m.get("content") or ""), m)
m = chat("What is the capital of Australia? One word.", **DIRECT); check("fact", "canberra" in (m.get("content") or "").lower(), m)
m = chat("Write a Python function is_prime(n). Code only.", max_tokens=400, **DIRECT); check("code", "def is_prime" in (m.get("content") or ""), m)
m = chat("Is 91 prime? Answer yes or no at the end.", max_tokens=2048, chat_template_kwargs={"enable_thinking": True})
think = m.get("reasoning") or m.get("reasoning_content") or ""
check("thinking", len(think) > 20 and re.search(r"\bno\b", (m.get("content") or "").lower()) is not None, m)
tools = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
m = chat("What's the weather in Paris right now?", tools=tools, tool_choice="auto")
tc = (m.get("tool_calls") or [{}])[0].get("function", {})
check("tool call", tc.get("name") == "get_weather" and "paris" in tc.get("arguments", "").lower(), m)
print(f"model {model}: " + ("ALL PASS" if ok else "SOME FAILED")); sys.exit(0 if ok else 1)
PY
