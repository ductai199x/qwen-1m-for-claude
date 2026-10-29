#!/usr/bin/env bash
# Verification phase 1 (TP=2, GPUs 0,1): is lookup drafting lossless, and is its copy-heavy gain real?
cd "$(dirname "$0")/../.."
export RESULTS=bench/verify/phase1.jsonl
S=(MODEL=ukisai/Swift-1.5-Qwen3.8-27b KV_OFFLOAD_GIB=0)
G=bench/bench_greedy.py; C=bench/bench_copy.py
BENCH="$G"             bench/run_config.sh V0-stock-nospec   "${S[@]}" SPEC_TOKENS=0                   -- --quantization fp8
BENCH="$G $C $C $C"    bench/run_config.sh V1-stock-dflash2  "${S[@]}" SPEC_METHOD=dflash              -- --quantization fp8
BENCH="$G $C $C $C"    bench/run_config.sh V2-port-lookup    "${S[@]}" SPEC_METHOD=dflash VLLM_BIN=.venv-port/bin/vllm VLLM_DFLASH2_LOOKUP=1 DRAFT_SAMPLE=probabilistic -- --quantization fp8
BENCH="$G $C"          bench/run_config.sh V3-port-nolookup  "${S[@]}" SPEC_METHOD=dflash VLLM_BIN=.venv-port/bin/vllm VLLM_DFLASH2_LOOKUP=0 DRAFT_SAMPLE=probabilistic -- --quantization fp8
