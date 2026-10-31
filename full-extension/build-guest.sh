#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/vllm-validation"
deadline=$((SECONDS + 1800))
until command -v clang++-18 >/dev/null && command -v cmake >/dev/null && command -v ninja >/dev/null; do
  if ((SECONDS >= deadline)); then
    echo 'Compiler provisioning did not complete within 30 minutes' >&2
    exit 2
  fi
  printf 'Waiting for provisioned compiler tools (%s seconds)\n' "$SECONDS"
  sleep 20
done
clang++-18 --version
cmake --version
wget --quiet --timeout=120 -O onednn-v3.13.tar.gz http://10.0.2.2:8765/onednn-v3.13.tar.gz
tar -xzf onednn-v3.13.tar.gz
cmake -S source -B build-base -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/clang-18 \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-18 \
  -DVLLM_TARGET_DEVICE=cpu \
  -DVLLM_RVV_VLEN=256 \
  -DVLLM_PYTHON_EXECUTABLE="$PWD/.venv/bin/python" \
  -DCMAKE_INSTALL_PREFIX="$PWD/source" \
  -DFETCHCONTENT_SOURCE_DIR_ONEDNN="$PWD/oneDNN-3.13"
cmake --build build-base --target _C --parallel 2
cmake --install build-base --component _C
