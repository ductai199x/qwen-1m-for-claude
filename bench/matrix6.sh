#!/usr/bin/env bash
# HyperQwen (vLLM 0.31.0 + patch series) within its tested context range: vLLM 0.31 overflows int32 indices at
# 1M context for this model (fixed in 0.32), so MAX_LEN=262144 and no 600K benchmark.
cd "$(dirname "$0")/.."
export RESULTS=bench/results8.jsonl BENCH="bench/bench_copy.py bench/bench_e2e.py"
S=(MODEL=ukisai/Swift-1.5-Qwen3.8-27b SPEC_METHOD=dflash KV_OFFLOAD_GIB=0 MAX_LEN=262144)
H=(VLLM_BIN=/home/tai/1-workdir/hyperqwen/venv/bin/vllm FLASHINFER_DISABLE_VERSION_CHECK=1 VLLM_USE_FLASHINFER_SAMPLER=0
   VLLM_SPEC_DECODE_ATTN=1 VLLM_DFLASH2_LOOKUP=1 DRAFT_SAMPLE=probabilistic)
bench/run_config.sh C-hyperqwen-k7-262k  "${S[@]}" "${H[@]}" VLLM_SPEC_DECODE_ATTN_QMAX=8 -- --quantization fp8
bench/run_config.sh D-hyperqwen-k15-262k "${S[@]}" "${H[@]}" SPEC_TOKENS=15 VLLM_SPEC_DECODE_ATTN_QMAX=16 -- --quantization fp8 --no-async-scheduling
