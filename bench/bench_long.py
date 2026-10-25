"""Long-context benchmark shaped like a Claude Code session.

1. cold:  ~N-token prompt, nothing cached         -> cold prefill time, decode tok/s at depth
2. warm:  same conversation + ~5K new tokens       -> warm TTFT (cache hit), decode tok/s at depth
3. short: short prompt                              -> decode tok/s at shallow depth

Usage: python bench_long.py PORT MODEL [TARGET_TOKENS] [SEED]
Prints one JSON line with the results.
"""
import json, random, sys, time, urllib.request

port, model = sys.argv[1], sys.argv[2]
target = int(sys.argv[3]) if len(sys.argv) > 3 else 600_000
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
URL = f"http://localhost:{port}"
rng = random.Random(seed)
CITIES = "Philadelphia Boston Denver Austin Seattle Chicago Miami Portland Atlanta Phoenix".split()
ITEMS = "bolts gears valves pumps sensors cables filters motors brackets hoses".split()


def records(start, n):
    return "\n".join(
        f"Record {i}: {rng.randint(1, 500)} {rng.choice(ITEMS)} shipped from {rng.choice(CITIES)} "
        f"to {rng.choice(CITIES)} on day {rng.randint(1, 365)}, status {rng.choice(['ok', 'late', 'lost'])}."
        for i in range(start, start + n))


def stream(messages, max_tokens):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "ignore_eos": True,
            "temperature": 1.0, "top_p": 0.95, "top_k": 20, "stream": True,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time(); t_first = None; usage = None; text = []
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                piece = d.get("content") or d.get("reasoning") or d.get("reasoning_content")
                if piece:
                    if t_first is None:
                        t_first = time.time()
                    if d.get("content"):
                        text.append(d["content"])
    t_end = time.time()
    n = usage["completion_tokens"]
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    return {"prompt": usage["prompt_tokens"], "cached": cached, "ttft": round(t_first - t0, 2),
            "decode_tps": round((n - 1) / (t_end - t_first), 1), "text": "".join(text)}


# Real source code as the long context (like a Claude Code session full of file reads), ~3.9 chars/token.
import glob, os
src = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "../.venv/lib/python3.12/site-packages/vllm/**/*.py"), recursive=True))
rng.shuffle(src)
files, size = [], 0
for f in src:
    body = open(f, errors="ignore").read()
    files.append(f"=== {os.path.relpath(f, os.path.dirname(f) + '/../..')} ===\n{body}")
    size += len(body)
    if size >= target * 3.9:
        break
doc = "\n\n".join(files)
import statistics
extras = [open(f, errors="ignore").read() for f in src[len(files):] if 12_000 < os.path.getsize(f) < 25_000][:5]
q1 = "Here is part of a Python codebase. Explain how the scheduler decides which requests to run next, citing the relevant functions."
msgs = [{"role": "user", "content": doc + "\n\n" + q1}]
cold = stream(msgs, 512)
warm = []
# Claude Code-like turns: each appends the previous answer plus a newly read file (~5K tokens)
for extra in extras:
    msgs += [{"role": "assistant", "content": (warm[-1] if warm else cold).pop("text")},
             {"role": "user", "content": "Here is another file:\n" + extra + "\n\nWrite a refactored version of its main class with type hints."}]
    warm.append(stream(msgs, 512))
warm[-1].pop("text"); cold.pop("text", None)
short = [stream([{"role": "user", "content": "Write a detailed explanation of how TCP congestion control works."}], 1024)["decode_tps"]
         for _ in range(3)]
med = lambda xs: round(statistics.median(xs), 2)
print(json.dumps({"cold": cold,
                  "warm": {"ttft_med": med([w["ttft"] for w in warm]), "decode_med": med([w["decode_tps"] for w in warm]),
                           "cached_frac": round(sum(w["cached"] or 0 for w in warm) / sum(w["prompt"] for w in warm), 3),
                           "prompt_last": warm[-1]["prompt"], "ttft": [w["ttft"] for w in warm], "decode": [w["decode_tps"] for w in warm]},
                  "short": {"decode_med": med(short), "decode": short}}))
