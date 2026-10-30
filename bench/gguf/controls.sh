#!/usr/bin/env bash
# Controls for compare.py: score llama.cpp's DavidAU greedy outputs with OTHER models in vLLM (GPU 1).
cd "$(dirname "$0")/../.."
for spec in "base Qwen/Qwen3.8-27B-FP8" "swift ukisai/Swift-1.5-Qwen3.8-27b"; do
    set -- $spec; name=$1; model=$2
    CUDA_VISIBLE_DEVICES=1 setsid .venv/bin/vllm serve $model --served-model-name $name --port 8021 --max-model-len 32768 \
        --gpu-memory-utilization 0.85 > bench/gguf/vllm-$name.log 2>&1 < /dev/null &
    pid=$!
    until curl -sf -o /dev/null localhost:8021/health || ! kill -0 $pid 2>/dev/null; do sleep 5; done
    .venv/bin/python -I bench/gguf/compare.py 8020 8021 $name models/davidau-hf bench/gguf/compare-control-$name.json 300 2>&1 | grep SUMMARY
    kill -TERM -- -$pid; for _ in $(seq 40); do kill -0 $pid 2>/dev/null || break; sleep 2; done; kill -KILL -- -$pid 2>/dev/null
    until [[ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1) -lt 1000 ]]; do sleep 2; done
done
