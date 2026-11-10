"""Single-agent repeat: bench_agents.py at concurrency 1, R times on one server; mean and sd of throughput and decode.
Usage: python bench_c1_repeat.py PORT MODEL [CTX] [R]"""
import json, os, statistics as st, subprocess, sys
port, model = sys.argv[1], sys.argv[2]
ctx = sys.argv[3] if len(sys.argv) > 3 and int(sys.argv[3]) < 300_000 else "100000"
R = int(sys.argv[4]) if len(sys.argv) > 4 else 5
here = os.path.dirname(os.path.abspath(__file__))
runs = []
for _ in range(R):
    out = subprocess.run([sys.executable, "-I", os.path.join(here, "bench_agents.py"), port, model, "1", ctx, "4"],
                         capture_output=True, text=True).stdout.strip().splitlines()
    runs.append(json.loads(out[-1]))
f = lambda k: [r[k] for r in runs]
ms = lambda xs: [round(st.mean(xs), 1), round(st.stdev(xs), 1)]
print(json.dumps({"runs": R, "agg_tok_s": ms(f("agg_tok_s")), "turn_decode_tps_med": ms(f("turn_decode_tps_med")),
                  "out_tokens": ms(f("out_tokens")), "per_run_agg": f("agg_tok_s")}))
