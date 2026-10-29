#!/usr/bin/env bash
# Phase 3a repeats on GPU 1 (same configs as phase3-attn.sh, run on the other card): sampler0 and both.
cd "$(dirname "$0")/../.."
export RESULTS=bench/verify/phase3-attn.jsonl RUN_GPUS=1 RUN_PORT=8013 BENCH=bench/bench_long.py
B=(MODEL=Qwen/Qwen3.8-27B-FP8 TP=1 GPU_UTIL=0.9 MAX_LEN=700000 KV_OFFLOAD_GIB=0 SPEC_METHOD=dflash
   VLLM_BIN=.venv-port/bin/vllm VLLM_DFLASH2_LOOKUP=0 DRAFT_SAMPLE=probabilistic SERVED_NAME=qwen-27b-1m)
for cfg in "sampler0 VLLM_SPEC_DECODE_ATTN=0 VLLM_USE_FLASHINFER_SAMPLER=0" \
           "both VLLM_SPEC_DECODE_ATTN=1 VLLM_SPEC_DECODE_ATTN_QMAX=8 VLLM_USE_FLASHINFER_SAMPLER=0"; do
    set -- $cfg; name=$1; shift
    for t in 100000 600000; do TARGET=$t bench/run_config.sh P3g1-$name-$((t/1000))k "${B[@]}" "$@"; done
done
echo phase3-attn-g1-done
