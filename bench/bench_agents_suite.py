"""Run bench_agents.py at concurrency 1, 4 and 10 against one server; print one JSON line with all three.
Usage: python bench_agents_suite.py PORT MODEL [CTX_TOKENS]"""
import json, os, subprocess, sys
port, model = sys.argv[1], sys.argv[2]
ctx = sys.argv[3] if len(sys.argv) > 3 and int(sys.argv[3]) < 300_000 else "100000"
here = os.path.dirname(os.path.abspath(__file__))
res = {}
for c in (1, 4, 10):
    out = subprocess.run([sys.executable, "-I", os.path.join(here, "bench_agents.py"), port, model, str(c), ctx, "4"],
                         capture_output=True, text=True).stdout.strip().splitlines()
    res[f"c{c}"] = json.loads(out[-1]) if out else {"error": "no output"}
print(json.dumps(res))
