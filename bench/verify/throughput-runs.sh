#!/usr/bin/env bash
# Throughput experiments for the live config (Swift 1.5 + DFlash2 + lookup, TP=2, 1M), concurrent-agent benchmark.
cd "$(dirname "$0")/../.."
export RESULTS=bench/verify/throughput.jsonl BENCH=bench/bench_agents_suite.py RUN_GPUS=0,1 RUN_PORT=8010 IDLE_MIB=6000 TARGET=100000
S=(MODEL=ukisai/Swift-1.5-Qwen3.8-27b SPEC_METHOD=dflash KV_OFFLOAD_GIB=0 VLLM_DFLASH2_LOOKUP=1 DRAFT_SAMPLE=probabilistic)
PORT_VENV=(VLLM_BIN=.venv-port/bin/vllm); HQ=(VLLM_BIN=/home/tai/1-workdir/hyperqwen-main/venv/bin/vllm)
bench/run_config.sh A0-live-baseline   "${S[@]}" "${PORT_VENV[@]}"                    -- --quantization fp8
bench/run_config.sh A1-k15-adaptive    "${S[@]}" "${PORT_VENV[@]}" SPEC_TOKENS=15     -- --quantization fp8 --no-async-scheduling
bench/run_config.sh A2-dynamic-k       "${S[@]}" "${PORT_VENV[@]}" \
    DYNAMIC_K_JSON='[[1,3,7],[4,8,3],[9,1024,1]]'                                     -- --quantization fp8
bench/run_config.sh B0-hqmain-baseline "${S[@]}" "${HQ[@]}"                           -- --quantization fp8
bench/run_config.sh B1-hqmain-int8pf   "${S[@]}" "${HQ[@]}" VLLM_PREFILL_ATTN=int8    -- --quantization fp8
echo throughput-runs-done
