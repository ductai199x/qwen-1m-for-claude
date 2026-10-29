#!/usr/bin/env bash
# Phase 2, GPU 0: HyperQwen's own field-test protocol on one H200 (README "Field tests wanted"), setup B
# (SPEC=dflash2 CTX=fast PREFIX_CACHE=1, the .env.example default), then the same with SPEC_ATTN=0.
set -u
OUT=/home/tai/1-workdir/02-vllm/bench/verify; cd /home/tai/1-workdir/hyperqwen
export PATH=$PWD/venv/bin:$PATH
serve() {  # name, env...
    local name=$1; shift; local log=$OUT/laneA-$name.server.log
    env CUDA_VISIBLE_DEVICES=0 PORT=18020 SPEC=dflash2 CTX=fast PREFIX_CACHE=1 "$@" setsid bash single-user/start_qwen.sh > "$log" 2>&1 &
    SPID=$!
    until curl -sf -o /dev/null localhost:18020/health || grep -qE 'Traceback|EngineDeadError|illegal memory|ILLEGAL_ADDRESS|refusing' "$log" || ! kill -0 $SPID 2>/dev/null; do sleep 5; done
    curl -sf -o /dev/null localhost:18020/health && echo "[$name] server up" || { echo "[$name] SERVER FAILED: $(grep -E 'Error|refusing' "$log" | tail -2)"; return 1; }
}
stop() { kill -TERM -- -$SPID 2>/dev/null; for _ in $(seq 60); do kill -0 $SPID 2>/dev/null || break; sleep 1; done; kill -KILL -- -$SPID 2>/dev/null
         until [[ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0) -lt 1000 ]]; do sleep 2; done; }

if serve setupB; then
    bash bench/run_benchmarks.sh single > $OUT/laneA-setupB-run1.txt 2>&1   # discard: first run after start reads low
    bash bench/run_benchmarks.sh single > $OUT/laneA-setupB-run2.txt 2>&1   # keep
    venv/bin/python bench/quality_battery.py h200-setupB --gsm-only > $OUT/laneA-setupB-gsm8k.txt 2>&1
    stop
fi
if serve setupB-specattn0 SPEC_ATTN=0; then
    bash bench/run_benchmarks.sh single > $OUT/laneA-specattn0-run1.txt 2>&1
    bash bench/run_benchmarks.sh single > $OUT/laneA-specattn0-run2.txt 2>&1
    stop
fi
echo laneA-done
