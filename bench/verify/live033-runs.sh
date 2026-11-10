#!/usr/bin/env bash
# vLLM 0.33 (.venv-033) vs 0.32 (.venv-port) with the exact live config: Swift 1.5 FP8, TP=2, 1M, DFlash2 + lookup,
# 200 GiB RAM tier, GPU_UTIL 0.78. GPUs 0+5. Features, agents at 100K, evicted-600K restore, and a ~980K request.
cd "$(dirname "$0")/../.."
export RESULTS=bench/verify/live033.jsonl RUN_GPUS=0,5 RUN_PORT=8010 IDLE_MIB=6000
S=(MODEL=ukisai/Swift-1.5-Qwen3.8-27b SPEC_METHOD=dflash KV_OFFLOAD_GIB=200 VLLM_DFLASH2_LOOKUP=1 DRAFT_SAMPLE=probabilistic GPU_UTIL=0.78)
FULL="bench/check_features.py bench/bench_agents_suite.py bench/bench_evict_check.py"
BENCH="$FULL" TARGET=600000 bench/run_config.sh E1-033-live "${S[@]}" VLLM_BIN=.venv-033/bin/vllm -- --quantization fp8
BENCH=bench/bench_long.py TARGET=980000 bench/run_config.sh E2-033-1m "${S[@]}" VLLM_BIN=.venv-033/bin/vllm -- --quantization fp8
BENCH="$FULL" TARGET=600000 bench/run_config.sh E3-032-live "${S[@]}" VLLM_BIN=.venv-port/bin/vllm -- --quantization fp8
BENCH=bench/bench_long.py TARGET=980000 bench/run_config.sh E4-032-1m "${S[@]}" VLLM_BIN=.venv-port/bin/vllm -- --quantization fp8
echo live033-runs-done
