# vLLM RISC-V numerical reproducers

These isolated correctness probes support PRs #64078 and #64079. They do not
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

The PR additionally includes 18 persistent pytest cases for NaNs, infinities,
signed zero, both even-tie directions, subnormal rounding, and finite overflow.
They compile the same production header. On the base header: 9 failed, 9 passed;
on the fixed header: 18 passed. This is helper coverage; the public CPU operator
test and model inference still require the target build and runtime.

## Remaining submission requirements

Full vLLM CPU operator regressions, real-weight model evaluation on the changed
RISC-V path, upstream CI, and the submitting human's review/test confirmation
are separate requirements. These probes alone do not satisfy them.

Prepared with OpenAI Codex assistance. The three upstream PRs remain drafts.
