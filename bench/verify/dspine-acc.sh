#!/usr/bin/env bash
# GSM8K accuracy, DFlash2 vs DSpine on the same vLLM 0.33 build and FP8 target (GPU 5, TP=1): is DSpine lossless?
cd "$(dirname "$0")/../.."
export RESULTS=bench/verify/dspine.jsonl BENCH=bench/bench_gsm8k_acc.py RUN_GPUS=5 RUN_PORT=8010 IDLE_MIB=6000 GSM8K_N=150
B=(MODEL=Qwen/Qwen3.8-27B-FP8 TP=1 MAX_LEN=131072 GPU_UTIL=0.85 KV_OFFLOAD_GIB=0 SPEC_METHOD=dflash VLLM_BIN=.venv-033/bin/vllm)
bench/run_config.sh A1-033-dflash2 "${B[@]}" DFLASH_MODEL=models/Qwen3.8-27B-DFlash2-1M SPEC_TOKENS=7
bench/run_config.sh A2-033-dspine  "${B[@]}" DFLASH_MODEL=tencent/Qwen3.8-27B-DSpine-b16 SPEC_TOKENS=15
echo dspine-acc-done
