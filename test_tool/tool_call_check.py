#!/usr/bin/env python3
# tool_call_check_v2.py
# Usage: set API_BASE, API_KEY, MODEL, then run: python tool_call_check_v2.py

import requests, json, re, time, sys

API_BASE = "http://127.0.0.1:3000/v1"   # 改成你的 base
API_KEY  = "sk-cQP8tIYLgFgznhoaEbC1Fc35878c42F6A84061101aCdF665"
MODEL    = "gemini-2.5-pro-preview-06-05-thinking"                           # 要检测的模型名
TIMEOUT  = 30

URL = API_BASE.rstrip('/') + "/chat/completions"

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

def post(payload):
    try:
        r = requests.post(URL, headers=HEADERS, json=payload, timeout=TIMEOUT)
        try:
            j = r.json()
        except Exception:
            j = None
        return r.status_code, r.text, j
    except Exception as e:
        return None, str(e), None

def inspect_response_json(j):
    """返回 (status, detail)：
       status ∈ {"CALLED","SUPPORTED_BUT_NOT_CALLED","SUPPORTED_ANNOUNCED","NO_TOOL_CALL_DETECTED","ERROR"}
    """
    if not isinstance(j, dict):
        return "ERROR", "no-json"
    if "error" in j:
        return "ERROR", j["error"]

    # top-level hints
    for key in ("tools","available_tools","tool_list"):
        if key in j:
            return "SUPPORTED_ANNOUNCED", { "field": key, "value": j[key] }

    choices = j.get("choices", [])
    # iterate choices for common patterns
    for c in choices:
        # Some APIs embed a 'message' object
        msg = c.get("message") or {}
        # 1) direct tool call objects (various possible keys)
        for key in ("tool_calls","tool_call","function_call"):
            # could be under message or under choice
            val = None
            if isinstance(msg, dict):
                val = msg.get(key)
            if val is None:
                val = c.get(key)
            if val:
                return "CALLED", {"how": key, "value": val}

        # 2) sometimes the assistant returns a 'tool' role message in a list (rare here)
        # check nested messages list if present
        if isinstance(j.get("messages"), list):
            for m in j["messages"]:
                if m.get("role") == "tool":
                    return "CALLED", {"role_tool_message": m}

        # 3) textual hint: assistant says it can use tools
        content = ""
        if isinstance(msg, dict):
            content = (msg.get("content") or "") 
        elif isinstance(c.get("delta") or "", dict):
            content = c.get("delta", {}).get("content", "")

        if isinstance(content, str) and re.search(
            r"\b(can|able to|able) (use|call|invoke) (tools|the|the `?ping`? tool|ping tool)\b", content, re.I):
            return "SUPPORTED_BUT_NOT_CALLED", {"content_snippet": content[:300]}

    # last resort: check completion-level keys
    for key in ("tool_calls","function_call","tool_call"):
        if key in j:
            return "CALLED", {"top_level": j[key]}

    return "NO_TOOL_CALL_DETECTED", {}

def make_test_payload(prompt_text, include_tools=True, ask_to_call=False, tool_choice="auto"):
    payload = {
        "model": MODEL,
        "messages": [
            {"role":"user", "content": prompt_text}
        ],
    }
    if include_tools:
        payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": "ping",
                    "description": "simple ping test tool",
                    "parameters": {"type":"object","properties":{}}
                }
            }
        ]
        # try to be compatible; some providers use tool_choice or function_call
        payload["tool_choice"] = tool_choice
    if ask_to_call:
        # stronger wording to encourage actual invocation
        payload["messages"][0]["content"] = ("Please call the ping tool now with no parameters. "
                                             "If you can call it, return a tool call (not plain text).")
    return payload

def pretty_print_final(status, detail, resp_json):
    print("\n--- Conclusion ---")
    if status == "CALLED":
        print("✅ Tool was invoked by the model (CALLED).")
        print("Detail:", json.dumps(detail, indent=2, ensure_ascii=False))
    elif status == "SUPPORTED_BUT_NOT_CALLED":
        print("ℹ️ Model claims it can use tools, but no tool-call object was returned.")
        print("Detail sample:", detail)
    elif status == "SUPPORTED_ANNOUNCED":
        print("ℹ️ API returned a tools/available_tools list (announced support).")
        print("Detail:", json.dumps(detail, indent=2, ensure_ascii=False))
    elif status == "NO_TOOL_CALL_DETECTED":
        print("❌ No evidence of tool calling in this response.")
        print("Possible reasons: provider doesn't expose tool-calling, model chosen doesn't support it, or extra flags are required.")
    else:
        print("❌ Error or unexpected response:", detail)
    # always show a compact response summary for debugging
    print("\n---- Response summary (compact) ----")
    try:
        print(json.dumps(resp_json if resp_json is not None else {"raw_text": "see above"}, indent=2, ensure_ascii=False)[:4000])
    except Exception:
        print("Unable to pretty print JSON.")

def main():
    # 1) initial permissive probe
    first = make_test_payload("Tool calling permission test: please invoke ping tool if available.", include_tools=True, ask_to_call=False)
    code, text, j = post(first)
    if code is None:
        print("Request failed:", text); return
    if code != 200:
        print(f"HTTP {code} returned. Body:\n{text}")
        return
    status, detail = inspect_response_json(j)
    if status == "NO_TOOL_CALL_DETECTED":
        # 2) follow-up: explicitly ask model to call the ping tool
        second = make_test_payload("", include_tools=True, ask_to_call=True, tool_choice="auto")
        code2, text2, j2 = post(second)
        if code2 is None:
            print("Second request failed:", text2); return
        if code2 != 200:
            print(f"Second request HTTP {code2}. Body:\n{text2}")
            # still show initial result
            pretty_print_final(status, detail, j)
            return
        status2, detail2 = inspect_response_json(j2)
        # prefer definitive result from second call
        if status2 in ("CALLED","SUPPORTED_BUT_NOT_CALLED","SUPPORTED_ANNOUNCED"):
            pretty_print_final(status2, detail2, j2)
            return
        else:
            # ambiguous: report both
            print("First response didn't show tool-call. Second probe also ambiguous.")
            pretty_print_final(status, detail, j)
            print("\n-- Second response (ambiguous) --")
            try:
                print(json.dumps(j2, indent=2, ensure_ascii=False)[:4000])
            except:
                print(text2[:4000])
            return
    else:
        pretty_print_final(status, detail, j)
        return

if __name__ == "__main__":
    main()