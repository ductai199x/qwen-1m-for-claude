#!/usr/bin/env bash
# DSpine (our vLLM 0.33 port of vllm#63704 + 27B sliding-window drafts) vs DFlash2, and vLLM 0.33 vs 0.32.
# One H200 (GPU 5), TP=1, 128K, base Qwen3.8-27B-FP8 (what both drafters were trained against), RAM tier off.
cd "$(dirname "$0")/../.."
export RESULTS=bench/verify/dspine.jsonl BENCH="bench/bench_accept.py bench/bench_agents_suite.py" RUN_GPUS=5 RUN_PORT=8010 IDLE_MIB=6000 TARGET=30000
B=(MODEL=Qwen/Qwen3.8-27B-FP8 TP=1 MAX_LEN=131072 GPU_UTIL=0.85 KV_OFFLOAD_GIB=0 SPEC_METHOD=dflash)
DF2=(DFLASH_MODEL=models/Qwen3.8-27B-DFlash2-1M SPEC_TOKENS=7)
bench/run_config.sh D1-033-dflash2        "${B[@]}" VLLM_BIN=.venv-033/bin/vllm "${DF2[@]}"
bench/run_config.sh D2-033-dspine         "${B[@]}" VLLM_BIN=.venv-033/bin/vllm DFLASH_MODEL=tencent/Qwen3.8-27B-DSpine-b16 SPEC_TOKENS=15
bench/run_config.sh D3-033-dflash2-lookup "${B[@]}" VLLM_BIN=.venv-033/bin/vllm "${DF2[@]}" VLLM_DFLASH2_LOOKUP=1 DRAFT_SAMPLE=probabilistic
bench/run_config.sh D4-032-dflash2-lookup "${B[@]}" VLLM_BIN=.venv-port/bin/vllm "${DF2[@]}" VLLM_DFLASH2_LOOKUP=1 DRAFT_SAMPLE=probabilistic
echo dspine-runs-done
