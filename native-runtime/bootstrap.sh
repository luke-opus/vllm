#!/usr/bin/env bash
set -euo pipefail

task_root="$1"
payload="$2"
mkdir -p "$task_root"/{logs,toolkit,downloads,results}
printf 'stage\texit_status\n' > "$task_root/results/stages.tsv"

run_stage() {
  local name="$1" status
  shift
  printf 'Starting stage: %s (%s)\n' "$name" "$(date -u +%FT%TZ)"
  printf '%q ' "$@" > "$task_root/logs/$name-command.txt"
  printf '\n' >> "$task_root/logs/$name-command.txt"
  if "$@" > "$task_root/logs/$name.txt" 2>&1; then status=0; else status=$?; fi
  printf '%s\t%s\n' "$name" "$status" >> "$task_root/results/stages.tsv"
  printf 'Finished stage: %s, actual exit %s (%s)\n' "$name" "$status" "$(date -u +%FT%TZ)"
  if [[ "$status" != 0 ]]; then cat "$task_root/logs/$name.txt"; return "$status"; fi
}

download() {
  local url="$1" path="$2"
  curl --http1.1 --fail --location --silent --show-error --retry 3 \
    --retry-delay 3 --connect-timeout 30 --max-time 1200 "$url" -o "$path"
}

[[ "$(uname -m)" == riscv64 ]]
run_stage apt-update sudo -n apt-get -o Acquire::Retries=3 update
run_stage apt-install sudo -n env DEBIAN_FRONTEND=noninteractive apt-get \
  -o Acquire::Retries=3 install -y --no-install-recommends build-essential \
  g++-13 cmake ninja-build pkg-config git curl ca-certificates unzip util-linux \
  python3.12 python3.12-dev python3.12-venv libnuma-dev libopenblas-dev libssl-dev

uv_wheel="$task_root/downloads/uv-0.13.6-py3-none-manylinux_2_31_riscv64.whl"
run_stage download-uv download \
  https://files.pythonhosted.org/packages/8d/c9/a412c2aa0494b307fce8c9c8234827f2fedb888954d3b3c8930a32a047c9/uv-0.13.6-py3-none-manylinux_2_31_riscv64.whl \
  "$uv_wheel"
printf '8542dcf118ce8a7cf2e4ba1fea7f5cc1638bc02b5b30dab1ead7e9034dc62fd3  %s\n' "$uv_wheel" | sha256sum -c -
unzip -q "$uv_wheel" -d "$task_root/toolkit/uv-package"
uv="$task_root/toolkit/uv-package/uv-0.13.6.data/scripts/uv"
chmod u+x "$uv"
export UV_CACHE_DIR="$task_root/uv-cache"
export UV_CONCURRENT_BUILDS=1
run_stage venv "$uv" venv --python /usr/bin/python3.12 "$task_root/.venv"
py="$task_root/.venv/bin/python"

export RUSTUP_HOME="$task_root/rustup"
export CARGO_HOME="$task_root/cargo"
export CARGO_BUILD_JOBS=1
run_stage rustup-download download \
  https://static.rust-lang.org/rustup/dist/riscv64gc-unknown-linux-gnu/rustup-init \
  "$task_root/toolkit/rustup-init"
run_stage rustup-digest-download download \
  https://static.rust-lang.org/rustup/dist/riscv64gc-unknown-linux-gnu/rustup-init.sha256 \
  "$task_root/toolkit/rustup-init.sha256"
rustup_digest="$(awk '{print $1}' "$task_root/toolkit/rustup-init.sha256")"
[[ "$rustup_digest" =~ ^[0-9a-f]{64}$ ]]
printf '%s  %s\n' "$rustup_digest" "$task_root/toolkit/rustup-init" | sha256sum -c -
chmod u+x "$task_root/toolkit/rustup-init"
run_stage rustup-install "$task_root/toolkit/rustup-init" -y --no-modify-path \
  --profile minimal --default-toolchain stable
export PATH="$task_root/.venv/bin:$task_root/cargo/bin:$PATH"
run_stage rust-version rustc --version --verbose
run_stage cmake-version cmake --version
run_stage compiler-version g++-13 --version

run_stage source-clone git -c http.version=HTTP/1.1 clone --filter=blob:none \
  --no-checkout https://github.com/vllm-project/vllm.git "$task_root/source"
run_stage source-fetch git -c http.version=HTTP/1.1 -C "$task_root/source" fetch \
  --filter=blob:none https://github.com/luke-opus/vllm.git \
  dfdc85947fe914d04906f5d0206c020c208bd400 e0d00e4b8eb0c89ec30921d0695cfdd409e9c211
run_stage source-checkout git -C "$task_root/source" checkout --detach \
  dfdc85947fe914d04906f5d0206c020c208bd400
[[ "$(git -C "$task_root/source" rev-parse HEAD)" == dfdc85947fe914d04906f5d0206c020c208bd400 ]]
git -C "$task_root/source" diff --exit-code \
  e0d00e4b8eb0c89ec30921d0695cfdd409e9c211 HEAD -- csrc cmake

torch_wheel="$task_root/downloads/torch-2.14.0+cpu-cp312-cp312-manylinux_2_39_riscv64.whl"
run_stage torch-download download \
  https://github.com/riseproject-dev/python-wheels/releases/download/torch-v2.14.0%2Bcpu-20260905085853/torch-2.14.0%2Bcpu-cp312-cp312-manylinux_2_39_riscv64.whl \
  "$torch_wheel"
printf '23da9af03ed3d293814721f0e9fdf50d5f49d1c4dfd922c6b547829d78783c12  %s\n' "$torch_wheel" | sha256sum -c -
run_stage python-build-deps "$uv" pip install --python "$py" \
  --index https://pypi.riseproject.dev/simple --index-strategy unsafe-best-match \
  "$torch_wheel" setuptools==78.1.1 'setuptools-scm>=8' 'setuptools-rust>=1.9.0' \
  'packaging>=24.2' wheel 'jinja2>=3.1.6' regex
run_stage runtime-deps "$uv" pip install --python "$py" \
  --index https://pypi.riseproject.dev/simple --index-strategy unsafe-best-match \
  -r "$task_root/source/requirements/cpu.txt"
run_stage additional-runtime-deps "$uv" pip install --python "$py" \
  --index https://pypi.riseproject.dev/simple --index-strategy unsafe-best-match \
  llguidance==1.7.6 \
  'https://github.com/riseproject-dev/python-wheels/releases/download/xgrammar-v0.2.8-20260925101607/xgrammar-0.2.8-cp312-cp312-manylinux_2_39_riscv64.whl#sha256=1128a32353783fb3624d51272f2ba9c0e86ca0e72d2681d09b43ba78133f325d' \
  'https://github.com/riseproject-dev/python-wheels/releases/download/apache-tvm-ffi-v0.1.15.post1-20261028103232/apache_tvm_ffi-0.1.15.post1-cp312-cp312-manylinux_2_39_riscv64.whl#sha256=6d3943a0d20635b37fb73a264f1423d83969a1062fbb7daed5914a5bbc6b3c59'

export VLLM_TARGET_DEVICE=cpu
export MAX_JOBS=1
export CC=gcc-13
export CXX=g++-13
export CMAKE_ARGS='-DVLLM_RVV_VLEN=0 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON'
# System CMake is real and meets >=3.26.1; no distribution metadata is invented.
run_stage editable-build "$uv" pip install --python "$py" --no-build-isolation \
  --index https://pypi.riseproject.dev/simple --index-strategy unsafe-best-match \
  -e "$task_root/source"
run_stage pip-check "$uv" pip check --python "$py"
run_stage pip-freeze "$uv" pip freeze --python "$py"
run_stage native-worker "$py" "$payload/native-openmp-pair.py" --execute \
  --source "$task_root/source" --output "$task_root/worker-results" \
  --model-files "$payload/model-files.json" --observer "$payload/check-native-openmp.py"
