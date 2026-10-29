"""Copy-heavy decode benchmark: the model rewrites a file from far back in its context.

Shaped like Claude Code applying an edit: ~CONTEXT tokens of source files, then "output this earlier
file in full with a small change". Most output tokens are a verbatim copy of text tens of thousands
of tokens back -- outside a drafter's sliding window, inside a context-lookup drafter's reach.

Usage: python bench_copy.py PORT MODEL [CONTEXT_TOKENS]
"""
import glob, json, os, random, statistics, sys, time, urllib.request

port, model = sys.argv[1], sys.argv[2]
context = int(sys.argv[3]) if len(sys.argv) > 3 else 100_000
URL = f"http://localhost:{port}"
src = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "../.venv/lib/python3.12/site-packages/vllm/**/*.py"), recursive=True))
random.Random(11).shuffle(src)
targets = [f for f in src if 8_000 < os.path.getsize(f) < 14_000][:3]   # ~2.5-4K-token files to rewrite
others = [f for f in src if f not in targets]


def stream(messages):
    body = {"model": model, "messages": messages, "max_tokens": 8000, "temperature": 1.0, "top_p": 0.95,
            "top_k": 20, "reasoning_effort": "low", "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); t_first = None; usage = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            usage = ev.get("usage") or usage
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                if t_first is None and (d.get("content") or d.get("reasoning") or d.get("reasoning_content")):
                    t_first = time.time()
    n = usage["completion_tokens"]
    return {"tokens": n, "ttft": round(t_first - t0, 2), "decode_tps": round((n - 1) / (time.time() - t_first), 1)}


rows = []
for i, target in enumerate(targets):
    # target file near the start, then filler code up to the context size (~3.9 chars/token)
    parts, size = [f"=== {os.path.basename(target)} ===\n" + open(target, errors="ignore").read()], 0
    for f in others[i * 400:]:
        body = open(f, errors="ignore").read(); parts.append(f"=== {os.path.basename(f)} ===\n{body}"); size += len(body)
        if size >= context * 3.9:
            break
    ask = (f"Output the complete file {os.path.basename(target)} from the start of this conversation, with one change: "
           "add a module-level comment `# reviewed` as the first line. Reproduce everything else exactly, in one code block.")
    rows.append(stream([{"role": "user", "content": "\n\n".join(parts) + "\n\n" + ask}]))
print(json.dumps({"decode_med": statistics.median(r["decode_tps"] for r in rows), "rows": rows}))
