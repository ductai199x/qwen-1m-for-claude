"""Find the longest prompt a server handles: send increasing-length code prompts until one fails.

Usage: python bench_lengths.py PORT MODEL
"""
import glob, json, os, random, sys, time, urllib.request
port, model = sys.argv[1], sys.argv[2]
src = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "../.venv/lib/python3.12/site-packages/vllm/**/*.py"), recursive=True))
random.Random(7).shuffle(src)
text = ""
for f in src:
    text += open(f, errors="ignore").read()
    if len(text) > 1_000_000 * 3.9:
        break
res = {}
for n in [32_000, 128_000, 200_000, 262_000, 300_000, 400_000, 600_000, 900_000]:
    body = {"model": model, "max_tokens": 256, "messages": [{"role": "user", "content": text[:int(n * 3.9)] + "\n\nSummarize."}]}
    t = time.time()
    try:
        r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://localhost:{port}/v1/chat/completions",
                      json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1800))
        res[n] = {"ok": True, "s": round(time.time() - t, 1), "prompt": r["usage"]["prompt_tokens"]}
    except Exception as e:
        res[n] = {"ok": False, "err": str(e)[:80]}
        break
print(json.dumps(res))
