#!/usr/bin/env bash
# HyperQwen's DFlash2 lookup + offload/GDN patches ported to vLLM 0.32 (.venv-port), full 1M context.
cd "$(dirname "$0")/.."
export RESULTS=bench/results8.jsonl
S=(MODEL=ukisai/Swift-1.5-Qwen3.8-27b SPEC_METHOD=dflash)
P=(VLLM_BIN=.venv-port/bin/vllm VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_SPEC_DECODE_ATTN=1 VLLM_DFLASH2_LOOKUP=1 DRAFT_SAMPLE=probabilistic)
BENCH="bench/bench_copy.py bench/bench_long.py bench/bench_e2e.py bench/bench_evict.py" \
bench/run_config.sh E-port-k7-1m  "${S[@]}" "${P[@]}" KV_OFFLOAD_GIB=200 VLLM_SPEC_DECODE_ATTN_QMAX=8 -- --quantization fp8
BENCH="bench/bench_copy.py bench/bench_e2e.py" \
bench/run_config.sh F-port-k15-1m "${S[@]}" "${P[@]}" KV_OFFLOAD_GIB=0 SPEC_TOKENS=15 VLLM_SPEC_DECODE_ATTN_QMAX=16 -- --quantization fp8 --no-async-scheduling
