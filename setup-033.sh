#!/usr/bin/env bash
# Build .venv-033: vLLM 0.33.0 with, in order,
#   1-dspine              Tencent DSpine drafting (vllm#63704) ported to 0.33, plus Qwen3.8-27B's sliding-window drafts
#   2-hyperqwen-port      HyperQwen's 0.33 exports of the topics .venv-port carries (lookup drafting, n-gram chains,
#                         GDN spec bounds, KV grouping, offload fixes incl. #424, env knobs) + our qwen3 parser fix
#   3-dspine-with-lookup  DSpine keeps the base DFlash propose() next to HyperQwen's lookup-drafting DFlash2
# The live .venv-port (0.32) is not touched. Usage: ./setup-033.sh  (idempotent)
set -euo pipefail
cd "$(dirname "$0")"
[[ -x .venv-033/bin/python ]] || uv venv -q --python 3.12 .venv-033
uv pip install -q --python .venv-033/bin/python "vllm==0.33.0"
# 0.33 needs FlashInfer's precompiled kernels installed separately, or the first start JIT-builds them (~20 min)
.venv-033/bin/vllm download-kernels
SP=$(.venv-033/bin/python -c 'import vllm, os; print(os.path.dirname(vllm.__file__))' 2>/dev/null | tail -n1)
for PATCH in "$PWD"/patches/vllm-0.33.0-*.patch; do
    if patch -p1 -R --dry-run -s --fuzz 0 -d "$SP" -i "$PATCH" >/dev/null 2>&1; then
        echo "already applied: $(basename "$PATCH")"
    else
        patch -p1 --forward --fuzz 0 --no-backup-if-mismatch -d "$SP" -i "$PATCH"
        echo "applied: $(basename "$PATCH")"
    fi
done
