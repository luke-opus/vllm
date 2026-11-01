#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0

if [[ $# != 2 && $# != 3 ]]; then
  echo "Usage: bash run.sh VLLM_GIT_REPO OUTPUT_DIR [TORCH_INCLUDE for ERF-only]" >&2
  exit 2
fi
repo="$(cd "$1" && pwd)"
mkdir -p "$2"
result_dir="$(cd "$2" && pwd)"
script_dir="$(cd "$(dirname "$0")" && pwd)"
base=ca8dbccf204abfcdf9dcb178f7e108c06b224b0f
erf=7f1353f1a5854545163d49f764552a57f836b2b2
bf16=174e56854703c0b9ede89e0a41f9e04dc4c59bc1
cxx_bf16="${CXX_BF16:-g++-13}"
cxx_rvv="${CXX_RVV:-clang++-18}"

[[ "$(uname -s)" == Linux && "$(uname -m)" == riscv64 ]]
printf 'stage\tvariant\texit_status\n' > "$result_dir/status.tsv"
uname -a > "$result_dir/uname.txt"
cat /proc/cpuinfo > "$result_dir/cpuinfo.txt"
: > "$result_dir/source-manifest.txt"

export_header() {
  local ref="$1" name="$2" folder="$3"
  mkdir -p "$folder"
  git -C "$repo" show "$ref:csrc/cpu/$name" > "$folder/$name"
  {
    printf '%s %s ' "$ref" "$name"
    git -C "$repo" rev-parse "$ref:csrc/cpu/$name"
    sha256sum "$folder/$name"
  } >> "$result_dir/source-manifest.txt"
}

run_binary() {
  local stage="$1" variant="$2" binary="$3" status
  if timeout 180 "$binary" > "$result_dir/$stage-$variant.txt" 2>&1; then
    status=0
  else
    status=$?
  fi
  printf '%s\t%s\t%s\n' "$stage" "$variant" "$status" >> "$result_dir/status.tsv"
  cat "$result_dir/$stage-$variant.txt"
  observed_status="$status"
}

if [[ $# == 2 ]]; then
command -v "$cxx_bf16" >/dev/null
"$cxx_bf16" --version > "$result_dir/compiler-bf16.txt"
for variant in base fixed; do
  ref="$base"
  if [[ "$variant" == fixed ]]; then ref="$bf16"; fi
  include="$result_dir/bf16-$variant"
  export_header "$ref" float_convert.hpp "$include"
  "$cxx_bf16" -std=c++20 -O2 -Wall -Wextra -Werror \
    -I "$include" "$script_dir/convert_test.cpp" \
    -o "$include/test" > "$result_dir/bf16-$variant-build.txt" 2>&1
  run_binary bf16 "$variant" "$include/test"
  if [[ "$variant" == base ]]; then
    [[ "$observed_status" == 1 ]]
    grep -Fx 'Checked 1458753 conversions; mismatches=667067' "$result_dir/bf16-base.txt"
  else
    [[ "$observed_status" == 0 ]]
    grep -Fx 'Checked 1458753 conversions; mismatches=0' "$result_dir/bf16-fixed.txt"
  fi
done

  echo 'BF16-only header comparison completed; no ERF or Torch coverage claimed.'
  exit 0
fi
torch_include="$(cd "$3" && pwd)"
[[ -f "$torch_include/torch/csrc/api/include/torch/all.h" ]]
command -v "$cxx_rvv" >/dev/null
"$cxx_rvv" --version > "$result_dir/compiler-erf.txt"

for variant in base fixed; do
  ref="$base"
  if [[ "$variant" == fixed ]]; then ref="$erf"; fi
  include="$result_dir/erf-$variant"
  for header in cpu_types_riscv.hpp cpu_types_riscv_defs.hpp cpu_types_riscv_impl.hpp float_convert.hpp; do
    export_header "$ref" "$header" "$include"
  done
  "$cxx_rvv" -std=c++20 -O2 -Wall -Wextra -ffp-contract=off \
    -march=rv64gcv_zvfh_zvl256b -mabi=lp64d -mrvv-vector-bits=256 \
    -I "$include" -isystem "$torch_include" \
    -isystem "$torch_include/torch/csrc/api/include" \
    "$script_dir/erf_header_test.cpp" -o "$include/test" \
    > "$result_dir/erf-$variant-build.txt" 2>&1
  readelf -d "$include/test" > "$result_dir/erf-$variant-dynamic.txt"
  run_binary erf "$variant" "$include/test"
  if [[ "$variant" == fixed ]]; then
    [[ "$observed_status" == 0 ]]
    grep -Fx 'checks=544 numerical_failures=0 nan_failures=0 sign_failures=0' "$result_dir/erf-fixed.txt"
  else
    [[ "$observed_status" == 0 || "$observed_status" == 1 ]]
  fi
done

echo 'Header comparisons completed; inspect base ERF output before claiming a reproduced silicon failure.'
