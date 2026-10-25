#!/usr/bin/env bash
# Benchmark one vLLM config on GPUs 0,1 with serve.sh.
#
# Usage: bench/run_config.sh NAME [ENV=VAL ...] [-- extra vllm serve args...]
# Appends a JSON line to ${RESULTS:-bench/results.jsonl}.
set -uo pipefail
cd "$(dirname "$0")/.."
name=$1; shift
envs=(); while (( $# )) && [[ $1 != -- ]]; do envs+=("$1"); shift; done; [[ ${1:-} == -- ]] && shift
PORT=8010; TARGET=${TARGET:-600000}
log=bench/logs/$name.log; mkdir -p bench/logs

env GPUS=0,1 PORT=$PORT "${envs[@]}" setsid ./serve.sh --enable-prompt-tokens-details "$@" > "$log" 2>&1 &
pid=$!
t0=$(date +%s)
until grep -q 'Application startup complete' "$log" || ! kill -0 $pid 2>/dev/null; do sleep 2; done
if ! kill -0 $pid 2>/dev/null; then
    echo "{\"name\": \"$name\", \"error\": \"failed to start\"}" | tee -a ${RESULTS:-bench/results.jsonl}
    grep -E 'Error|error' "$log" | tail -3
    exit 1
fi
startup=$(( $(date +%s) - t0 ))
kv=$(grep -oE 'GPU KV cache size: [0-9,]+' "$log" | grep -oE '[0-9,]+$' | tr -d ,)
mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0,1 | paste -sd/)
# BENCH may list several benchmark scripts; each gets its own result line
for bench in ${BENCH:-bench/bench_long.py}; do
    res=$(.venv/bin/python -I $bench $PORT qwen-27b-1m $( [[ $bench == *bench_e2e* ]] || echo "$TARGET" ) 2>&1 | tail -1)
    echo "{\"name\": \"$name\", \"bench\": \"$(basename $bench .py)\", \"startup_s\": $startup, \"kv_tokens\": ${kv:-null}, \"vram_mib\": \"$mem\", \"result\": ${res:-null}}" | tee -a ${RESULTS:-bench/results.jsonl}
done

# stop: SIGTERM the process group, then SIGKILL if the HTTP server hangs on open connections
kill -TERM -- -$pid 2>/dev/null
for _ in $(seq 60); do kill -0 $pid 2>/dev/null || break; sleep 1; done
kill -KILL -- -$pid 2>/dev/null
until [[ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0,1 | sort -n | tail -1) -lt 1000 ]]; do sleep 2; done
