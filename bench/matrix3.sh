#!/usr/bin/env bash
# Eviction scenario (4 x 600K sessions, return to the first) with and without CPU KV offloading.
cd "$(dirname "$0")/.."
export RESULTS=bench/results3.jsonl BENCH=bench/bench_evict.py
bench/run_config.sh evict-gpu-only     -- --max-num-batched-tokens 32768
bench/run_config.sh evict-cpu-offload  -- --max-num-batched-tokens 32768 --kv-offloading-size 200 --kv-offloading-backend native
