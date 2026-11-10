"""Feature check for a Claude Code backend: the request shapes Claude Code sends, plus images and the prefix cache.

  messages_tool    /v1/messages, effort "high", one tool -> a tool_use block with the right arguments
  messages_stream  same endpoint, streaming -> SSE message_start ... message_stop with text deltas
  inline_system    a system message mid-conversation (Claude Code's reminders) -> 200, not a template error
  image            /v1/chat/completions with a solid red PNG -> the answer names the colour
  prefix_cache     the same ~20K-token prompt twice -> the second hits the prefix cache for >= 90% of it (server /metrics)

Usage: python check_features.py PORT MODEL
"""
import base64, json, struct, sys, urllib.request, zlib

PORT, MODEL = sys.argv[1], sys.argv[2]
URL = f"http://localhost:{PORT}"
HDR = {"Content-Type": "application/json", "anthropic-version": "2023-06-01"}


def post(path, body, raw=False):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), HDR)
    with urllib.request.urlopen(req, timeout=600) as r:
        return r.read().decode() if raw else json.load(r)


def png(rgb, size=64):
    row = b"\x00" + bytes(rgb) * size
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(row * size)) + chunk(b"IEND", b""))


tool = {"name": "get_weather", "description": "Get the weather for a city",
        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}
res = {}
try:
    r = post("/v1/messages", {"model": MODEL, "max_tokens": 2000, "output_config": {"effort": "high"}, "tools": [tool],
                              "messages": [{"role": "user", "content": "What's the weather in Paris? Use the tool."}]})
    uses = [b for b in r.get("content", []) if b["type"] == "tool_use"]
    res["messages_tool"] = bool(uses) and uses[0]["name"] == "get_weather" and "paris" in json.dumps(uses[0]["input"]).lower()
except Exception as e: res["messages_tool"] = f"error: {e}"
try:
    s = post("/v1/messages", {"model": MODEL, "max_tokens": 200, "stream": True,
                              "messages": [{"role": "user", "content": "Say hello in five words."}]}, raw=True)
    res["messages_stream"] = "message_start" in s and "message_stop" in s and "text_delta" in s
except Exception as e: res["messages_stream"] = f"error: {e}"
try:
    r = post("/v1/messages", {"model": MODEL, "max_tokens": 300, "system": "You are a helpful coding agent.",
                              "messages": [{"role": "user", "content": "Remember the number 42."},
                                           {"role": "assistant", "content": "Got it."},
                                           {"role": "system", "content": "<system-reminder>Be brief.</system-reminder>"},
                                           {"role": "user", "content": "What number?"}]})
    res["inline_system"] = "42" in json.dumps(r.get("content", []))
except Exception as e: res["inline_system"] = f"error: {e}"
try:
    img = base64.b64encode(png((220, 20, 20))).decode()
    r = post("/v1/chat/completions", {"model": MODEL, "max_tokens": 300, "temperature": 0, "messages": [{"role": "user", "content": [
        {"type": "text", "text": "What single colour fills this image? One word."},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}}]}]})
    res["image"] = "red" in (r["choices"][0]["message"]["content"] or "").lower()
except Exception as e: res["image"] = f"error: {e}"
try:
    doc = " ".join(f"Line {i}: the quick brown fox number {i} jumps over lazy dog {i * 7}." for i in range(1500))
    body = {"model": MODEL, "max_tokens": 5, "messages": [{"role": "user", "content": doc + "\nReply OK."}]}
    def hits():   # server-side counter: works whether or not the server reports cached tokens per request
        t = urllib.request.urlopen(URL + "/metrics", timeout=60).read().decode()
        return sum(float(l.split()[-1]) for l in t.splitlines() if l.startswith("vllm:prefix_cache_hits"))
    post("/v1/chat/completions", body); h0 = hits(); u = post("/v1/chat/completions", body)["usage"]
    frac = (hits() - h0) / u["prompt_tokens"]
    res["prefix_cache"] = round(frac, 3) >= 0.9; res["prefix_cache_frac"] = round(frac, 3); res["prompt_tokens"] = u["prompt_tokens"]
except Exception as e: res["prefix_cache"] = f"error: {e}"
res["all_pass"] = all(v is True for k, v in res.items() if k in ("messages_tool", "messages_stream", "inline_system", "image", "prefix_cache"))
print(json.dumps(res))
