#!/usr/bin/env bash
# A0 (our port) vs B0 (HyperQwen main), same flags, interleaved repeats to separate a real difference from noise.
cd "$(dirname "$0")/../.."
export RESULTS=bench/verify/throughput.jsonl BENCH=bench/bench_agents_suite.py RUN_GPUS=0,1 RUN_PORT=8010 IDLE_MIB=6000 TARGET=100000
S=(MODEL=ukisai/Swift-1.5-Qwen3.8-27b SPEC_METHOD=dflash KV_OFFLOAD_GIB=0 VLLM_DFLASH2_LOOKUP=1 DRAFT_SAMPLE=probabilistic)
for rep in 2 3; do
    bench/run_config.sh A0-live-baseline-r$rep   "${S[@]}" VLLM_BIN=.venv-port/bin/vllm                            -- --quantization fp8
    bench/run_config.sh B0-hqmain-baseline-r$rep "${S[@]}" VLLM_BIN=/home/tai/1-workdir/hyperqwen-main/venv/bin/vllm -- --quantization fp8
done
echo repeats-done
