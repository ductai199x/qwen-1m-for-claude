#!/usr/bin/env bash
# HyperQwen #424 (retain SWA history at sparse Mamba offload boundaries) on the #405 H200/600K eviction workload.
# Same as offload-runs.sh R1-R4, with the patch applied in overlays/{hq424,port424}/vllm (PYTHONPATH shadows the venv).
cd "$(dirname "$0")/../.."
export RESULTS=bench/verify/offload.jsonl BENCH=bench/bench_evict2.py RUN_GPUS=0,1 RUN_PORT=8010 IDLE_MIB=6000
B=(MODEL=Qwen/Qwen3.8-27B-FP8 KV_OFFLOAD_GIB=200 SERVED_NAME=qwen-27b-1m)
HQ=(VLLM_BIN=/home/tai/1-workdir/hyperqwen-main/venv/bin/vllm PYTHONPATH=$PWD/overlays/hq424 VLLM_USE_FLASHINFER_SAMPLER=0)
PORT=(VLLM_BIN=.venv-port/bin/vllm PYTHONPATH=$PWD/overlays/port424)
bench/run_config.sh P3-hqmain424-dflash2 "${B[@]}" "${HQ[@]}"   SPEC_METHOD=dflash VLLM_DFLASH2_LOOKUP=1 DRAFT_SAMPLE=probabilistic
bench/run_config.sh P4-hqmain424-mtp     "${B[@]}" "${HQ[@]}"   SPEC_METHOD=mtp
bench/run_config.sh P1-port424-dflash2   "${B[@]}" "${PORT[@]}" SPEC_METHOD=dflash VLLM_DFLASH2_LOOKUP=1 DRAFT_SAMPLE=probabilistic
echo offload-pr424-done
