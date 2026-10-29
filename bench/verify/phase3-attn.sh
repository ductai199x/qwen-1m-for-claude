#!/usr/bin/env bash
# Phase 3a, GPU 0, TP=1: which HyperQwen knob slows long-context decode on H200, and how it scales with context.
# Port (.venv-port), official Qwen/Qwen3.8-27B-FP8 + DFlash2 (RoPE-fixed), lookup OFF in all four, so only the two
# knobs vary. bench_long at 100K and 600K (5 warm turns each -> medians).
cd "$(dirname "$0")/../.."
export RESULTS=bench/verify/phase3-attn.jsonl RUN_GPUS=0 RUN_PORT=8012 BENCH=bench/bench_long.py
B=(MODEL=Qwen/Qwen3.8-27B-FP8 TP=1 GPU_UTIL=0.9 MAX_LEN=700000 KV_OFFLOAD_GIB=0 SPEC_METHOD=dflash
   VLLM_BIN=.venv-port/bin/vllm VLLM_DFLASH2_LOOKUP=0 DRAFT_SAMPLE=probabilistic SERVED_NAME=qwen-27b-1m)
for cfg in "neither VLLM_SPEC_DECODE_ATTN=0" \
           "specattn VLLM_SPEC_DECODE_ATTN=1 VLLM_SPEC_DECODE_ATTN_QMAX=8" \
           "sampler0 VLLM_SPEC_DECODE_ATTN=0 VLLM_USE_FLASHINFER_SAMPLER=0" \
           "both VLLM_SPEC_DECODE_ATTN=1 VLLM_SPEC_DECODE_ATTN_QMAX=8 VLLM_USE_FLASHINFER_SAMPLER=0"; do
    set -- $cfg; name=$1; shift
    for t in 100000 600000; do TARGET=$t bench/run_config.sh P3-$name-$((t/1000))k "${B[@]}" "$@"; done
done
echo phase3-attn-done
