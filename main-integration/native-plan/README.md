# Current-main Cloud-V header plan (inactive)

These scripts and the workflow candidate are **prepared and inactive**. No native
run, guest run, compilation, or model test was performed for this preparation.
Keep the candidate at `main-integration/native-plan/workflow.candidate.yml` in
the evidence branch; that path does not activate GitHub Actions. The original
`cloud-v-header/` evidence is unchanged. Wait for the separate machine inventory
and verify the required ISA before scheduling a native run.

## Exact production inputs

| Input | Commit |
| --- | --- |
| Common base | `e0d00e4b8eb0c89ec30921d0695cfdd409e9c211` |
| BF16 fixed | `6592612016059859dff1e93bd534edc1a5737615` |
| ERF fixed | `ec0fbf5e9b3e9d796ccae667a437f2e97c3f9cde` |

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
  e0d00e4b8eb0c89ec30921d0695cfdd409e9c211 \
  ec0fbf5e9b3e9d796ccae667a437f2e97c3f9cde \
  6592612016059859dff1e93bd534edc1a5737615
```

After verifying and extracting those Torch headers into `torch-headers`, run the
two phases from the evidence branch root:

```sh
CXX_BF16=g++-13 bash main-integration/native-plan/run.sh source-refs results-bf16
CXX_RVV=clang++-18 bash main-integration/native-plan/run.sh \
  source-refs results-erf torch-headers/torch/include
```

For a later scheduled run, put the candidate at
`.github/workflows/cloud-v-header.yml` only on `luke-opus/vllm` branch
`riscv-cloud-v-headers`. Retain `run.sh`, `convert_test.cpp` and
`erf_header_test.cpp` at `main-integration/native-plan/` in that same commit.
The candidate downloads exactly that path from the run's `GITHUB_SHA`; do not
move those payload files to `cloud-v-header/`. Before activation, verify all
three source commits above are published and fetchable from the user fork.
This preparation did not copy the candidate into `.github/workflows/`, push a
branch, dispatch a workflow, install dependencies, or contact a runner.
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

## Preparation evidence and limits

[Header identity comparison](header-blob-comparison.json) records old and new
commit IDs, Git blob IDs and SHA256 values for four production headers across
base, ERF-fixed and BF16-fixed: **12 of 12 comparisons are byte-identical**.
Both C++ test files and the custom-runner actionlint config are unchanged copies.
The earlier fixed-ERF cross-compile used these same four production headers and
the same test source; the existing local assembly inspection recorded compiler
exit 0 and retained `e16` `vfadd.vv` for the Zvfh probe. That historical result
covers identical input content and was not rerun or relabeled as a current-main
execution. It does not establish the planned native Clang 18 link or execution.

[Preparation checks](preparation-checks.txt) records only newly performed local
Bash syntax, YAML/payload-reference and existing-tool actionlint checks, with
actual exit statuses. The original [preparation log](../../cloud-v-header/preparation-checks.txt)
remains historical and unchanged. No earlier static check is counted as a new
run. No old runner inventory/status file is copied into this plan.

To repeat the workflow static check from the evidence branch root using an
already available actionlint binary:

```sh
actionlint -config-file main-integration/native-plan/actionlint.yaml \
  main-integration/native-plan/workflow.candidate.yml
bash -n main-integration/native-plan/run.sh
```

The config only declares the provider's custom `banana-pi-f3` label. Runner labels,
CPU metadata and success inside a container do not independently prove physical
silicon. Header results do not establish model accuracy, full extension/serving
correctness, performance, upstream CI, human review, or independent physical
hardware attestation. The original VLEN-256 and actual Zvfh requirements remain;
this candidate is not a VLEN-128 validation plan.
