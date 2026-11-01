# Prepared Cloud-V header comparisons

These scripts and the workflow candidate are **prepared, not executed on Cloud-V**.
Keep the candidate as `cloud-v-header/workflow.candidate.yml` in this evidence branch: that
path does not activate GitHub Actions. Wait for the separate machine inventory
before scheduling a native run.

## Exact production inputs

| Input | Commit |
| --- | --- |
| Common base | `ca8dbccf204abfcdf9dcb178f7e108c06b224b0f` |
| BF16 fixed | `174e56854703c0b9ede89e0a41f9e04dc4c59bc1` |
| ERF fixed | `7f1353f1a5854545163d49f764552a57f836b2b2` |

`run.sh` exports the original headers independently from each commit and records
Git blob IDs and SHA256 hashes. BF16 uses production `float_convert.hpp`; ERF
includes the full production `cpu_types_riscv.hpp` and directly exercises
`vec_op::FP32Vec8/16::er()`. No extracted algorithm, invented Vec/Torch type, or
modified mask/tail policy is used.

## Dependencies and execution

The candidate targets Ubuntu 24.04 RISC-V Linux, VLEN 256 and actual Zvfh vector
arithmetic. It records machine information first and installs only missing
`g++-13 clang-18 git curl ca-certificates unzip binutils coreutils` packages.
GCC 13 is used for scalar BF16; Clang 18 compiles ERF with
`-march=rv64gcv_zvfh_zvl256b -mrvv-vector-bits=256 -mabi=lp64d`.

ERF requires real Torch headers because the production header includes
`<torch/all.h>`. The candidate downloads the [RISE Torch wheel](https://github.com/riseproject-dev/python-wheels/releases/download/torch-v2.14.0%2Bcpu-20260905085853/torch-2.14.0%2Bcpu-cp312-cp312-manylinux_2_39_riscv64.whl),
verifies **133,517,449 bytes** and SHA256
`23da9af03ed3d293814721f0e9fdf50d5f49d1c4dfd922c6b547829d78783c12`,
and extracts only `torch/include/*`. No Torch installation, Python import, Torch
shared library, vLLM extension build, or model download is required by this plan.

Fetch the three commits; explicit promisor configuration lets `git show` retrieve
the original blobs omitted by the filtered fetch:

```sh
git init source-refs
git -C source-refs remote add origin https://github.com/luke-opus/vllm.git
git -C source-refs config remote.origin.promisor true
git -C source-refs config remote.origin.partialclonefilter blob:none
git -C source-refs fetch --filter=blob:none --depth=1 origin \
  ca8dbccf204abfcdf9dcb178f7e108c06b224b0f \
  7f1353f1a5854545163d49f764552a57f836b2b2 \
  174e56854703c0b9ede89e0a41f9e04dc4c59bc1
```

After verifying and extracting those Torch headers into `torch-headers`, run the
two phases from the directory containing this folder:

```sh
CXX_BF16=g++-13 bash cloud-v-header/run.sh source-refs results-bf16
CXX_RVV=clang++-18 bash cloud-v-header/run.sh \
  source-refs results-erf torch-headers/torch/include
```

For a later scheduled run, put the candidate at
`.github/workflows/cloud-v-header.yml` only on `luke-opus/vllm` branch
`riscv-cloud-v-headers`, alongside the three scripts in `cloud-v-header/`.
Its branch-limited push trigger avoids changing the default branch; it also
declares `workflow_dispatch`. The single job has `contents: read`, a 45-minute
limit, and retrieves scripts at that run's exact `GITHUB_SHA`, without JavaScript
checkout/upload actions. The exit trap prints raw setup, compilation, execution,
manifest and status logs from fixed result directories while retaining failures.

## Interpretation

- BF16 requires 1,458,753 checks: base exit 1 with 667,067 mismatches, fixed exit 0
  with zero mismatches, including the existing SiLU(0.125) bit check.
- ERF reads `vlenb`, executes half-vector addition, then requires fixed exit 0
  with 544 checks and zero numerical, NaN and sign failures. Finite comparisons
  use `1e-6 + 1e-5 * abs(std::erf(x))`; special inputs include signed zeros,
  infinities and NaNs.
- Native ERF base **may pass** because mask-agnostic lanes may remain unchanged.
  Preserve and report that result as “baseline failure not reproduced on this
  implementation”; do not force all-ones lanes or change the algorithm.

Local preparation passed Bash syntax and workflow actionlint checks. A real
header cross-compile succeeded; its assembly retained `e16` `vfadd.vv` in the
Zvfh probe. Native Clang 18 linking and execution remain unverified. Header
results do not establish model accuracy, full extension/serving correctness,
performance, upstream CI, human review, or independent physical-silicon attestation.

The original [preparation log](preparation-checks.txt) retains local execution
paths. To repeat its static workflow check from this evidence directory, use
`actionlint -config-file cloud-v-header/actionlint.yaml cloud-v-header/workflow.candidate.yml`.
The config declares the provider's custom `banana-pi-f3` runner label; other
workflow checks remain enabled.
