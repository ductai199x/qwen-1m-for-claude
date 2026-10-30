"""Concurrent coding-agent benchmark: C sessions at once, each a growing Claude-Code-like conversation.

Each session starts from its own ~CTX-token code context (distinct per session), then runs TURNS turns; every turn
appends the previous answer plus a newly "read" file (~2-4K tokens, like a Read tool result) and alternates between a
copy-heavy request (rewrite the file) and a reasoning request (review it). Thinking is on, sampling is the model's
recommended T=1.0/top_p=0.95/top_k=20. Sessions run in parallel threads; turns within a session are sequential.

Reports aggregate output tok/s (all generated tokens / wall time), per-turn decode tok/s and TTFT (medians), and
how many tokens came from the prefix cache.

Usage: python bench_agents.py PORT MODEL CONCURRENCY [CTX_TOKENS] [TURNS]
"""
import glob, json, os, random, statistics, sys, threading, time, urllib.request

port, model, conc = sys.argv[1], sys.argv[2], int(sys.argv[3])
ctx = int(sys.argv[4]) if len(sys.argv) > 4 else 100_000
turns = int(sys.argv[5]) if len(sys.argv) > 5 else 4
URL = f"http://localhost:{port}"
SRC = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "../.venv/lib/python3.12/site-packages/vllm/**/*.py"), recursive=True))
SMALL = [f for f in SRC if 8_000 < os.path.getsize(f) < 16_000]


def stream(messages):
    body = {"model": model, "messages": messages, "max_tokens": 4000, "temperature": 1.0, "top_p": 0.95, "top_k": 20,
            "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); t_first = None; usage = None; text = []
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:]); usage = ev.get("usage") or usage
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                if t_first is None and (d.get("content") or d.get("reasoning") or d.get("reasoning_content")):
                    t_first = time.time()
                if d.get("content"): text.append(d["content"])
    t_end = time.time(); n = usage["completion_tokens"]
    return {"n": n, "ttft": t_first - t0, "tps": (n - 1) / max(t_end - t_first, 1e-6), "prompt": usage["prompt_tokens"],
            "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0, "text": "".join(text)}


def session(i, out):
    rng = random.Random(1000 + i)
    files = SRC[:]; rng.shuffle(files)
    parts, size = [], 0
    for f in files:
        body = open(f, errors="ignore").read(); parts.append(f"=== {os.path.basename(f)} ===\n{body}"); size += len(body)
        if size >= ctx * 3.9: break
    msgs = [{"role": "system", "content": "You are a coding agent working in this repository."},
            {"role": "user", "content": "\n\n".join(parts) + "\n\nI'll show you files to work on next. Reply 'ready'."}]
    rows = [stream(msgs)]
    for t in range(turns):
        msgs.append({"role": "assistant", "content": rows[-1].pop("text")})
        f = SMALL[rng.randrange(len(SMALL))]; code = open(f, errors="ignore").read()
        ask = ("Rewrite this file with full type hints. Output the complete file in one code block." if t % 2 == 0 else
               "Review this file: explain its design and point out likely bugs, briefly.")
        msgs.append({"role": "user", "content": f"<tool_result file='{os.path.basename(f)}'>\n{code}\n</tool_result>\n\n{ask}"})
        rows.append(stream(msgs))
    rows[-1].pop("text", None)
    out[i] = rows


out = {}
t0 = time.time()
threads = [threading.Thread(target=session, args=(i, out)) for i in range(conc)]
for th in threads: th.start()
for th in threads: th.join()
wall = time.time() - t0
all_rows = [r for i in sorted(out) for r in out[i]]
turn_rows = [r for i in sorted(out) for r in out[i][1:]]          # exclude each session's cold first request
med = lambda xs: round(statistics.median(xs), 2) if xs else None
print(json.dumps({"concurrency": conc, "ctx": ctx, "turns": turns, "wall_s": round(wall, 1),
                  "out_tokens": sum(r["n"] for r in all_rows), "agg_tok_s": round(sum(r["n"] for r in all_rows) / wall, 1),
                  "turn_decode_tps_med": med([r["tps"] for r in turn_rows]), "turn_ttft_med": med([r["ttft"] for r in turn_rows]),
                  "cold_ttft_med": med([out[i][0]["ttft"] for i in out]),
                  "cached_frac": round(sum(r["cached"] for r in turn_rows) / max(sum(r["prompt"] for r in turn_rows), 1), 3)}))
