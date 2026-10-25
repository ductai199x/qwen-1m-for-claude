#!/usr/bin/env bash
# Upgrade the vllm venv to the latest versions.
#
# Usage: ./upgrade.sh [--no-upgrade]   (--no-upgrade: sync to the current uv.lock only)
#
# Env overrides:
#   CUDA_ARCHS   GPU arch(s) for the source builds, e.g. "90" (H200) or "80;90" (default: 90)
#   MAX_JOBS     parallel compile jobs (default: 32)
set -euo pipefail
cd "$(dirname "$0")"

CUDA_ARCHS="${CUDA_ARCHS:-90}"
export MAX_JOBS="${MAX_JOBS:-32}"
SOURCE_PKGS=(flash-attn causal-conv1d)

# 1. Refresh the lockfile.
if [[ "${1:-}" != "--no-upgrade" ]]; then
    cp uv.lock uv.lock.bak 2>/dev/null || true
    uv lock --upgrade
fi

# 2. Install everything except the source-built CUDA extensions, so the new
#    torch is in place before they compile (--no-build-isolation builds
#    against whatever torch is in the venv).
no_install=()
for p in "${SOURCE_PKGS[@]}"; do no_install+=(--no-install-package "$p"); done
uv sync --no-build-isolation "${no_install[@]}"

# 3. Build the extensions with g++ (uv's standalone Python defaults CXX to clang++).
#    Reinstall them so they're rebuilt against the new torch even if their version didn't change.
torch_archs=""  # "80;90;100" -> "8.0;9.0;10.0"
IFS=';' read -ra archs <<<"$CUDA_ARCHS"
for a in "${archs[@]}"; do torch_archs+="${torch_archs:+;}${a:0:-1}.${a: -1}"; done
# Also reinstall oss-harmony: it ships the same openai_harmony/ dir as the old openai-harmony,
# so uninstalling the latter during an upgrade deletes its files.
reinstall=(--reinstall-package oss-harmony)
for p in "${SOURCE_PKGS[@]}"; do reinstall+=(--reinstall-package "$p"); done
CC=gcc CXX=g++ CUDAHOSTCXX=g++ \
TORCH_CUDA_ARCH_LIST="$torch_archs" \
FLASH_ATTN_CUDA_ARCHS="$CUDA_ARCHS" \
NVCC_THREADS=2 \
    uv sync --no-build-isolation "${reinstall[@]}"

# 4. Smoke test.
.venv/bin/python - <<'EOF'
import torch, vllm, flash_attn, causal_conv1d, fla, transformers
import vllm.entrypoints.openai.api_server  # catches missing deps that `import vllm` alone doesn't
from flash_attn import flash_attn_func
from causal_conv1d import causal_conv1d_fn
q = torch.randn(1, 128, 8, 64, device="cuda", dtype=torch.bfloat16)
flash_attn_func(q, q, q)
x = torch.randn(2, 64, 128, device="cuda", dtype=torch.bfloat16)
causal_conv1d_fn(x, torch.randn(64, 4, device="cuda", dtype=torch.bfloat16))
print(f"OK  torch {torch.__version__} | vllm {vllm.__version__} | flash-attn {flash_attn.__version__} | "
      f"causal-conv1d {causal_conv1d.__version__} | fla {fla.__version__} | transformers {transformers.__version__}")
EOF
