#!/usr/bin/env bash
# Single-agent repeat (5x, 100K context), 0.33 vs 0.32, exact live config, GPUs 0+5: is 0.33's c1 gain real?
cd "$(dirname "$0")/../.."
export RESULTS=bench/verify/live033.jsonl RUN_GPUS=0,5 RUN_PORT=8010 IDLE_MIB=6000 BENCH=bench/bench_c1_repeat.py TARGET=600000
S=(MODEL=ukisai/Swift-1.5-Qwen3.8-27b SPEC_METHOD=dflash KV_OFFLOAD_GIB=200 VLLM_DFLASH2_LOOKUP=1 DRAFT_SAMPLE=probabilistic GPU_UTIL=0.78)
bench/run_config.sh C1-033 "${S[@]}" VLLM_BIN=.venv-033/bin/vllm -- --quantization fp8
bench/run_config.sh C1-032 "${S[@]}" VLLM_BIN=.venv-port/bin/vllm -- --quantization fp8
echo live033-c1-done
