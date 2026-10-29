#!/usr/bin/env bash
# Port diagnostics at 1M: why is decode at 600K 4x slower? Lookup scan (whole context per step) vs the
# split-KV Triton verify attention vs the forced torch top-k sampler.
cd "$(dirname "$0")/.."
export RESULTS=bench/results8.jsonl BENCH="bench/bench_copy.py bench/bench_long.py"
S=(MODEL=ukisai/Swift-1.5-Qwen3.8-27b SPEC_METHOD=dflash KV_OFFLOAD_GIB=0 VLLM_BIN=.venv-port/bin/vllm DRAFT_SAMPLE=probabilistic)
bench/run_config.sh G1-port-lookup-search64k "${S[@]}" VLLM_DFLASH2_LOOKUP=1 VLLM_DFLASH2_LOOKUP_SEARCH=65536 VLLM_SPEC_DECODE_ATTN=0 -- --quantization fp8
bench/run_config.sh G2-port-nolookup-specattn "${S[@]}" VLLM_DFLASH2_LOOKUP=0 VLLM_SPEC_DECODE_ATTN=1 VLLM_SPEC_DECODE_ATTN_QMAX=8 VLLM_USE_FLASHINFER_SAMPLER=0 -- --quantization fp8
