#!/usr/bin/env bash
# Drafter (DFlash2 vs MTP) and think-less fine-tune (Swift 1.5) experiments.
cd "$(dirname "$0")/.."
export RESULTS=bench/results4.jsonl BENCH="bench/bench_long.py bench/bench_e2e.py"
C="--max-num-batched-tokens 32768"
bench/run_config.sh mtp-fp8                                         -- $C
bench/run_config.sh dflash2-fp8          SPEC_METHOD=dflash         -- $C
BENCH=bench/bench_e2e.py bench/run_config.sh swift-mtp  MODEL=ukisai/Swift-1.5-Qwen3.8-27b -- $C --quantization fp8
BENCH=bench/bench_e2e.py bench/run_config.sh swift-dflash2 MODEL=ukisai/Swift-1.5-Qwen3.8-27b SPEC_METHOD=dflash -- $C --quantization fp8
