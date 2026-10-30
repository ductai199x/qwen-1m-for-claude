"""Hold a fixed amount of memory on some GPUs so they never look idle, then sleep until killed.

Used to hand GPUs between a long-lived holder and a benchmark without a window where an opportunistic job (another
user's "wait until the GPU is clear" script) can grab them. The allocation never changes, so it does not disturb vLLM's
startup memory profiling. Usage: python gpu_bridge.py GIB GPU [GPU ...]
"""
import sys, time, torch
gib, gpus = float(sys.argv[1]), [int(g) for g in sys.argv[2:]]
held = [torch.empty(int(gib * 2**30), dtype=torch.uint8, device=f"cuda:{g}") for g in gpus]
print(f"bridge holding {gib} GiB on GPUs {gpus}", flush=True)
while True:
    time.sleep(3600)
