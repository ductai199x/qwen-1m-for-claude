#!/usr/bin/env bash
# Phase 3b, GPU 1, TP=1: is greedy output reproducible on one server, and where does each speculative config
# first diverge from the no-spec reference? Official Qwen/Qwen3.8-27B-FP8; token ids saved per run.
cd "$(dirname "$0")/../.."
export RESULTS=bench/verify/phase3-greedy.jsonl RUN_GPUS=1 RUN_PORT=8013
B=(MODEL=Qwen/Qwen3.8-27B-FP8 TP=1 MAX_LEN=200000 KV_OFFLOAD_GIB=0 SERVED_NAME=qwen-27b-1m)
G=bench/bench_greedy.py
# same server twice: repeatability
BENCH="$G@a $G@b" bench/run_config.sh G-nospec "${B[@]}" SPEC_TOKENS=0
BENCH="$G@a $G@b" bench/run_config.sh G-stock-dflash2 "${B[@]}" SPEC_METHOD=dflash
BENCH="$G" GREEDY_TAG=-a bench/run_config.sh G-port-lookup "${B[@]}" SPEC_METHOD=dflash VLLM_BIN=.venv-port/bin/vllm VLLM_DFLASH2_LOOKUP=1 DRAFT_SAMPLE=probabilistic
BENCH="$G" GREEDY_TAG=-a bench/run_config.sh G-port-nolookup "${B[@]}" SPEC_METHOD=dflash VLLM_BIN=.venv-port/bin/vllm VLLM_DFLASH2_LOOKUP=0 DRAFT_SAMPLE=probabilistic
echo phase3-greedy-done
