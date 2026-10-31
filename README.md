# vLLM RISC-V numerical reproducers

These numerical probes and configuration regressions support PRs #64078, #64079, and #64080. They do not
build vLLM, load weights, measure performance, or validate a real RISC-V machine.
The production method bodies and headers are read from the requested Git refs.
No candidate fix is synthesized by the probes.

## Requirements

- Node.js (tested: 24.19.0; no npm packages required).
- A host C++20 compiler (tested: Apple clang 21).
- Zig with RISC-V vector intrinsics (tested: 0.17.0).
- `qemu-system-riscv64` with `rvv_ta_all_1s` and `rvv_ma_all_1s`
  (tested: 11.1.2).
- A local vLLM checkout containing the base and PR heads below.

Run from this directory. `CXX`, `ZIG`, and `QEMU` may each name an executable;
they default to `c++`, `zig`, and `qemu-system-riscv64`. Results and generated
sources are written under `results/`. The vLLM checkout is read only.

## ERF (#64078)

```sh
node make-probe.mjs /path/to/vllm \
  ca8dbccf204abfcdf9dcb178f7e108c06b224b0f \
  7f1353f1a5854545163d49f764552a57f836b2b2
node run-probes.mjs
node check-special.mjs
```

The generator extracts the complete production `exp()` and `er()` methods of
`FP32Vec8` and `FP32Vec16`. The minimal surrounding structs provide only
load/save operations and register storage; PyTorch and vLLM dispatch are absent.
The production RVV typedef header is copied from the fixed ref.

For VLEN 128 and 256, 256 inputs per vector class are compared to host `std::erf`.
QEMU runs both its default agnostic policy and a policy that fills agnostic lanes
with ones, which the ISA permits. Each fixed run must pass all 512 numerical
checks and preserve the signs of zero. The special-value probe adds signed zero,
infinities, NaNs, and +/-1 for both vector classes.

Observed results: each agnostic-ones baseline run reports 262 NaN failures out of
512 checks. Default baseline runs report two signed-zero failures. All four fixed
numerical runs and all four fixed special-value runs pass. The scripts fail if
the numerical reproducer does not reproduce the baseline NaNs or a fixed run
fails. A passing shell exit means the *comparison* is as expected, including the
deliberate baseline failures.

## Software BF16 (#64079)

```sh
node run-bf16.mjs /path/to/vllm \
  ca8dbccf204abfcdf9dcb178f7e108c06b224b0f \
  174e56854703c0b9ede89e0a41f9e04dc4c59bc1
```

This compiles the actual `float_convert.hpp` separately for each ref. The
reference chooses the nearest representable BF16 value using long-double
distances and even-tie selection, instead of repeating the production integer
bias expression. It checks six low-word boundaries for every high word, one
million deterministic inputs, all BF16 round trips, and SiLU(0.125).

Observed results: 1,458,753 conversions per ref; baseline 667,067 mismatches,
fixed zero. SiLU(0.125) changes from `0x3d87` to the expected `0x3d88`.

A prepared local follow-up, not yet pushed to PR #64079, adds 18 pytest cases for NaNs, infinities,
signed zero, both even-tie directions, subnormal rounding, and finite overflow.
They compile the same production header. On the base header: 9 failed, 9 passed;
on the fixed header: 18 passed. This is helper coverage; the public CPU operator
test and model inference still require the target build and runtime.

The same 18-case test was also compiled and executed inside Ubuntu 24.04
RISC-V Linux under QEMU 11.1.2, using GCC 13.3 and Python 3.12.3: baseline
9 failed/9 passed, fixed 18 passed. See `evidence/bf16-riscv-linux-pytest.txt`.
This validates the header on the emulated Linux target, not native silicon
or a built vLLM extension. To prepare a small offline-target test archive:

```sh
/path/to/.venv/bin/python make-bf16-pytest.py /path/to/vllm \
  ca8dbccf204abfcdf9dcb178f7e108c06b224b0f \
  174e56854703c0b9ede89e0a41f9e04dc4c59bc1
```

Transfer `results/bf16-pytest.tar.gz` to the target, extract it, and run the
following from the extracted directory with a `uv` virtual environment
containing pytest and an available C++20 compiler:

```sh
CXX=g++-13 /path/to/.venv/bin/python -m pytest --noconftest -q base/tests/kernels/core/test_cpu_float_conversion.py
CXX=g++-13 /path/to/.venv/bin/python -m pytest --noconftest -q fixed/tests/kernels/core/test_cpu_float_conversion.py
```

The baseline command must exit 1 with 9 failed/9 passed, and the fixed command
must exit 0 with 18 passed. The archive generator obtains both headers through
`git show`; it never constructs a candidate fix.

## OpenMP configuration (#64080)

Use a Python 3.12 environment containing the checkout's `requirements/common.txt`,
Torch 2.14.0, and pytest:

```sh
/path/to/.venv/bin/python run-openmp.py /path/to/vllm \
  ca8dbccf204abfcdf9dcb178f7e108c06b224b0f \
  018ef906c8cfc72cdfa21eafac157258c63373cb
```

The runner extracts each ref's real `vllm` package into separate temporary
directories, runs the PR's six configuration cases, and preserves the original
checkout. Observed: base 2 failed/4 passed, fixed 6 passed. Both failures are
missing bundled-libgomp additions. The test constructs real `VllmConfig` and
`DeviceConfig` objects and calls `CpuPlatform.check_and_update_config`. Host
OS/architecture, affinity, Torch's package location and temporary library files
are simulated external inputs. This is a configuration regression on macOS
arm64, not execution of a RISC-V worker or its OpenMP runtime.

## Original pre-commit configuration

All applicable hooks for each PR's changed files passed with the unmodified
repository configuration, Node LTS 26.13.0, and the pinned hook revisions.
The BF16 run includes the prepared local 18-case follow-up and CPU CI wiring.
See `evidence/precommit-*.txt`. Tools were bootstrapped using a locally cached
Node distribution checked against Node's official SHA256; no hooks were
disabled. These are local checks, not upstream CI results.

## Documentation builds

All three complete documentation builds passed the original MkDocs configuration
with API documentation generation enabled:

```sh
../docs-validation/.venv/bin/mkdocs build --strict --site-dir ../docs-validation/site-erf-full
../docs-validation/.venv/bin/mkdocs build --strict --site-dir ../docs-validation/site-bf16-full
../docs-validation/.venv/bin/mkdocs build --strict --site-dir ../docs-validation/site-openmp-full
```

Run each command from its corresponding PR checkout. Build times were 214.41,
204.82, and 204.23 seconds. See `evidence/docs-*-strict.txt`. Existing broken
architecture anchors reported at INFO level remain visible in the captured logs.
Local-checkout and GitHub documentation findability checks passed. The upstream
Read the Docs preview URLs returned HTTP 404; a successful local build does not
establish that an upstream preview or CI passed.

An independent [rendered CPU documentation preview](https://luke-opus.github.io/vllm/)
is available for all three original PR heads. Pages builds and HTTP 200 checks
passed, and a fresh agent read all three through Chrome and verified each
installation-index-to-CPU link. Its OpenMP question passed; GELU and BF16
questions partially passed because explicit default/activation wording was
missing. Clarifications are prepared locally and are not in those PR heads or
this preview yet. Web-search queries returned no indexed results, and the web
tool could not open the preview; Chrome was the working fallback.

## Remaining submission requirements

Additional [complete CPU extension comparisons](full-extension/README.md) now
reproduce and fix both numerical bugs in real registered GELU, SiLU and MoE
operators on emulated RISC-V Linux. They use independent full libraries and
document the identical BLAS/OpenMP/dependency adaptations. Repository pytest
and model-entry validation are separate from these operator comparisons.

Prepared [documentation/test/CI follow-ups](followups/) remain local candidates.
Their generated documentation previews are available under
[ERF candidate](https://luke-opus.github.io/vllm/64078-candidate/getting_started/installation/cpu/)
and [BF16 candidate](https://luke-opus.github.io/vllm/64079-candidate/getting_started/installation/cpu/).
The original upstream PR heads are unchanged.

[Runtime dependency records](runtime-dependencies/README.md) document real
RISC-V Harmony/XGrammar/LLGuidance builds and native API checks, plus the
observed repository/LLM import failures before the missing packages were
installed. The subsequent repository pytest and model runs are in progress.

[Current-main candidates](followups/latest-main/README.md) apply cleanly to
`e5196e2`; they are prepared patches, not rebuilt or tested current-main commits.
The original branches are 338 commits behind that main. The recorded upstream
pre-run failure is a contributor trust gate; current merged-PR count is zero.

Repository pytest, real-weight model evaluation on the changed
RISC-V path, upstream CI, and the submitting human's review/test confirmation
are separate requirements. These probes alone do not satisfy them.

Prepared with OpenAI Codex assistance. The three upstream PRs remain drafts.
