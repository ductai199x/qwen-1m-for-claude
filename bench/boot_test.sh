#!/usr/bin/env bash
# Boot one serve.sh config on GPUs 0,1 and report whether it reaches "ready" or crashes; then stop it.
# A crashed engine can leave the API server hanging, so crashes are detected from the log.
#
# Usage: bench/boot_test.sh NAME [ENV=VAL ...] [-- extra vllm serve args...]
set -uo pipefail
cd "$(dirname "$0")/.."
name=$1; shift
envs=(); while (( $# )) && [[ $1 != -- ]]; do envs+=("$1"); shift; done; [[ ${1:-} == -- ]] && shift
log=bench/logs/boot-$name.log; mkdir -p bench/logs
RUN_GPUS=${RUN_GPUS:-0,1}
env GPUS=$RUN_GPUS PORT=${RUN_PORT:-8010} "${envs[@]}" setsid ./serve.sh "$@" > "$log" 2>&1 &
pid=$!
pat='Traceback|EngineDeadError|illegal memory|ILLEGAL_ADDRESS|device-side assert|out of memory'
until grep -q 'Application startup complete' "$log" || grep -qE "$pat" "$log" || ! kill -0 $pid 2>/dev/null; do sleep 3; done
if grep -q 'Application startup complete' "$log"; then
    echo "$name: READY"
else
    echo "$name: CRASHED -- $(grep -E 'Error' "$log" | grep -v Warning | grep -vE '^\s*$' | tail -1 | sed -E 's/^.*\] //' | cut -c1-200)"
fi
kill -TERM -- -$pid 2>/dev/null
for _ in $(seq 30); do kill -0 $pid 2>/dev/null || break; sleep 1; done
kill -KILL -- -$pid 2>/dev/null
until [[ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $RUN_GPUS | sort -n | tail -1) -lt 1000 ]]; do sleep 2; done
