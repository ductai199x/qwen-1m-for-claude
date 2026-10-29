#!/usr/bin/env bash
# Phase 2, GPU 1 (TP=1): claim 3 (stock DFlash2 drafter past 262K; the max_position_embeddings fix), on the
# official Qwen/Qwen3.8-27B-FP8 target; then claim 1 (pristine vLLM 0.31.0 at 1M context, no spec decode).
cd "$(dirname "$0")/../.."
export RESULTS=bench/verify/laneB.jsonl RUN_GPUS=1 RUN_PORT=8011
Q=(MODEL=Qwen/Qwen3.8-27B-FP8 TP=1 KV_OFFLOAD_GIB=0 SPEC_METHOD=dflash MAX_LEN=400000 SERVED_NAME=qwen-27b-1m)
BENCH=bench/bench_lengths.py bench/run_config.sh B3-stock-drafter   "${Q[@]}" DFLASH_MODEL=incoai/Qwen3.8-27B-DFlash2
BENCH=bench/bench_lengths.py bench/run_config.sh B3-patched-drafter "${Q[@]}" DFLASH_MODEL=models/Qwen3.8-27B-DFlash2-1M
P=(MODEL=Qwen/Qwen3.8-27B-FP8 TP=1 KV_OFFLOAD_GIB=0 SPEC_TOKENS=0 GPU_UTIL=0.9 VLLM_BIN=/home/tai/1-workdir/hyperqwen/venv-pristine031/bin/vllm)
bench/boot_test.sh B1-pristine031-262k "${P[@]}" MAX_LEN=262144                           | tee -a bench/verify/laneB-boot.txt
bench/boot_test.sh B1-pristine031-1m   "${P[@]}" MAX_LEN=1000000                          | tee -a bench/verify/laneB-boot.txt
bench/boot_test.sh B1-pristine031-1m-blocking "${P[@]}" MAX_LEN=1000000 CUDA_LAUNCH_BLOCKING=1 | tee -a bench/verify/laneB-boot.txt
bench/boot_test.sh B1-stock032-1m      MODEL=Qwen/Qwen3.8-27B-FP8 TP=1 KV_OFFLOAD_GIB=0 SPEC_TOKENS=0 GPU_UTIL=0.9 MAX_LEN=1000000 | tee -a bench/verify/laneB-boot.txt
echo laneB-done
