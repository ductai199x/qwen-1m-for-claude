#!/usr/bin/env bash
# Offload-with-DFlash2 question from HyperQwen #405: R1/R2 our port, R3/R4 HyperQwen main (vLLM 0.32 + their series).
# Base Qwen/Qwen3.8-27B-FP8, TP=2, 1M (YaRN x4), extended-RoPE DFlash2 drafter, 200 GiB CPU tier, bench_evict2.py.
cd "$(dirname "$0")/../.."
export RESULTS=bench/verify/offload.jsonl BENCH=bench/bench_evict2.py RUN_GPUS=0,1 RUN_PORT=8010
B=(MODEL=Qwen/Qwen3.8-27B-FP8 KV_OFFLOAD_GIB=200 SERVED_NAME=qwen-27b-1m)
HQ=/home/tai/1-workdir/hyperqwen-main/venv/bin/vllm
bench/run_config.sh R1-port-dflash2  "${B[@]}" SPEC_METHOD=dflash VLLM_BIN=.venv-port/bin/vllm VLLM_DFLASH2_LOOKUP=1 DRAFT_SAMPLE=probabilistic
bench/run_config.sh R2-port-mtp      "${B[@]}" SPEC_METHOD=mtp    VLLM_BIN=.venv-port/bin/vllm
bench/run_config.sh R3-hqmain-dflash2 "${B[@]}" SPEC_METHOD=dflash VLLM_BIN=$HQ VLLM_DFLASH2_LOOKUP=1 DRAFT_SAMPLE=probabilistic VLLM_USE_FLASHINFER_SAMPLER=0
bench/run_config.sh R4-hqmain-mtp    "${B[@]}" SPEC_METHOD=mtp    VLLM_BIN=$HQ VLLM_USE_FLASHINFER_SAMPLER=0
echo offload-runs-done
