"""Greedy outputs for a fixed prompt set, to check that a speculative-decoding change is lossless.

Prints one JSON line: per prompt the sha256 of the generated token ids, so two servers can be compared
(identical hashes = identical greedy output). Copy-heavy prompts exercise the context lookup; the others
are ordinary generation.

Usage: python bench_greedy.py PORT MODEL [TOKENS_OUT.json]
"""
import glob, hashlib, json, os, random, sys, urllib.request

port, model = sys.argv[1], sys.argv[2]
tokens_out = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3].endswith(".json") else None
all_tokens = []
URL = f"http://localhost:{port}"
src = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "../.venv/lib/python3.12/site-packages/vllm/**/*.py"), recursive=True))
random.Random(5).shuffle(src)
files = [f for f in src if 6_000 < os.path.getsize(f) < 10_000][:3]
filler = "\n\n".join(open(f, errors="ignore").read() for f in src[100:160])[:120_000]   # ~30K tokens

prompts = []
for f in files:
    prompts.append(f"=== {os.path.basename(f)} ===\n{open(f, errors='ignore').read()}\n\n{filler}\n\n"
                   f"Output the file {os.path.basename(f)} from the start of this message exactly, in one code block.")
prompts += ["Explain how a B-tree insertion works, step by step.",
            "Write a Python function that parses ISO-8601 durations like P3DT4H, with tests.",
            "What are the tradeoffs between optimistic and pessimistic locking?"]

out = []
for p in prompts:
    body = {"model": model, "messages": [{"role": "user", "content": p}], "max_tokens": 1500, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}, "logprobs": True, "top_logprobs": 0}
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    toks = [t["token"] for t in r["choices"][0]["logprobs"]["content"]]
    all_tokens.append(toks)
    out.append({"n": len(toks), "sha": hashlib.sha256("\x00".join(toks).encode()).hexdigest()[:16],
                "head": r["choices"][0]["message"]["content"][:60]})
if tokens_out:
    json.dump(all_tokens, open(tokens_out, "w"))
print(json.dumps({"rows": out}))
