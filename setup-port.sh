#!/usr/bin/env bash
# Build .venv-port: the project's locked environment (vLLM 0.32.0) with patches/vllm-0.32.0-*.patch applied:
#   dflash2-lookup          DFlash2 context-lookup drafting ported from HyperQwen
#   offload-mamba-swa-replay  HyperQwen #424: CPU KV offload restores with DFlash2 (else 0 hits on every return)
# The live .venv is not touched.
#
# Usage: ./setup-port.sh        (idempotent: re-running on a patched tree reports it and does nothing)
set -euo pipefail
cd "$(dirname "$0")"

# flash-attn and causal-conv1d are source builds the server doesn't need (vLLM ships its own kernels)
UV_PROJECT_ENVIRONMENT=.venv-port uv sync --frozen --no-build-isolation \
    --no-install-package flash-attn --no-install-package causal-conv1d
SP=$(.venv-port/bin/python -c 'import vllm, os; print(os.path.dirname(vllm.__file__))' 2>/dev/null | tail -n1)

for PATCH in "$PWD"/patches/vllm-0.32.0-dflash2-lookup.patch "$PWD"/patches/vllm-0.32.0-offload-mamba-swa-replay.patch; do
    if patch -p1 -R --dry-run -s --fuzz 0 -d "$SP" -i "$PATCH" >/dev/null 2>&1; then
        echo "already applied: $(basename "$PATCH")"
    else
        patch -p1 --forward --fuzz 0 --no-backup-if-mismatch -d "$SP" -i "$PATCH"
        echo "applied: $(basename "$PATCH")"
    fi
done
