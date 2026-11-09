"""Speculative-decoding acceptance on GSM8K and HumanEval: how many tokens each verify step accepts.

N prompts per suite, one at a time, greedy (temperature 0), thinking on (the template's default), up to MAX_TOKENS
each. Acceptance length = 1 + accepted draft tokens / verify steps, from the server's spec_decode metrics around each
suite (the "+1" is the token the target model always contributes). Also reports decode tok/s per suite.

Usage: python bench_accept.py PORT MODEL [N] [MAX_TOKENS]
"""
import json, os, re, sys, time, urllib.request

PORT, MODEL = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 and int(sys.argv[3]) < 10000 else 40
MAX_TOKENS = int(sys.argv[4]) if len(sys.argv) > 4 else 1024
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
suites = {
    "gsm8k": [q + "\nSolve step by step and give the final answer."
              for q in json.load(open(f"{DATA}/gsm8k_test.json"))[:N]],
    "humaneval": ["Complete this Python function. Reply with the full function in one code block.\n\n" + q
                  for q in json.load(open(f"{DATA}/humaneval.json"))[:N]],
}


def metrics():
    t = urllib.request.urlopen(f"http://localhost:{PORT}/metrics", timeout=60).read().decode()
    get = lambda k: sum(float(v) for v in re.findall(rf"^vllm:{k}(?:_total)?(?:\{{[^}}]*\}})? ([0-9.e+-]+)$", t, re.M))
    return get("spec_decode_num_drafts"), get("spec_decode_num_accepted_tokens")


out = {}
for name, prompts in suites.items():
    d0, a0 = metrics(); toks = 0; dec_s = 0.0
    for p in prompts:
        body = {"model": MODEL, "messages": [{"role": "user", "content": p}], "max_tokens": MAX_TOKENS, "temperature": 0}
        t = time.time()
        r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://localhost:{PORT}/v1/chat/completions",
                      json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1800))
        dec_s += time.time() - t; toks += r["usage"]["completion_tokens"]
    d1, a1 = metrics()
    out[name] = {"accept_len": round(1 + (a1 - a0) / max(d1 - d0, 1), 3), "tok_s": round(toks / dec_s, 1), "tokens": toks}
print(json.dumps(out))
