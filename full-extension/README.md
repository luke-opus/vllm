# Complete CPU extension validation on emulated RISC-V Linux

This is additional software validation for the three draft PRs. It is not
native silicon, upstream CI, or the submitting human's review/test attestation.

The original base is `ca8dbccf204abfcdf9dcb178f7e108c06b224b0f`.
ERF uses `7f1353f1a5854545163d49f764552a57f836b2b2` and BF16 uses
`174e56854703c0b9ede89e0a41f9e04dc4c59bc1`. Each numerical fix is built
independently; the ERF build does not contain the BF16 fix and vice versa.

## Build and compatibility details

Ubuntu 24.04 riscv64 runs in QEMU 11.1.2 with RVV 1.0, VLEN 256, Zvfh,
software BF16, and both agnostic-lane policies set to ones. The guest account
has no sudo. Publisher checksums and exact dependency refs are recorded in
`validation-environment.json`.

Native Clang 18 CMake configuration generated 229 compilation units, including
oneDNN v3.13 at `0e2a5bfeef1bfbffc3137464606540233086ce9b`. Host Zig 0.17.0
replayed this complete list using the guest's actual Torch 2.14.0+cpu headers,
Linux/GCC 13 sysroot and generated attention header. All 229 objects compiled;
each independent fix then rebuilt all 19 production source objects. The actual
guest linker produced the complete CPU shared libraries, rather than isolated
method bodies. No compiled binaries or VM credentials are published here.

Three environment adaptations must be reproduced and are not upstream fixes:

1. oneDNN's `rvv_softmax.cpp` needs `std::isfinite` declared for the host
   compiler. `compat-math.hpp` imports the standard declaration; the original
   failure and successful command remain in the compilation record.
2. Clang-generated OpenMP symbols need Ubuntu's LLVM runtime loaded:
   `LD_PRELOAD=/usr/lib/llvm-18/lib/libomp.so`.
3. RISE's bundled OpenBLAS lacks `sbgemm_`. The default extension consequently
   fails to load. All three comparison builds use the existing real PyTorch
   reference BLAS fallback in `blas_gemm.h`: compile `fla.cpp` with
   `-UVLLM_HAS_OPENBLAS`. No dummy GEMM symbol, operator replacement or lazy
   linker bypass is used. The original `fla.cpp.o` is byte-identical in all
   three refs; its replacement is the same in all three comparison builds.

The native-link and full compilation commands are under `evidence/`. The
scripts preserve the original CMake commands and map only compiler, target and
header paths. Their host Zig path and mirror/sysroot directories must be set
for another host. `package-reference-blas.py` validates each original object
before replacing it, and archives/manifests contain SHA256 verification.

Complete linked library sizes:

| Build | Bytes |
| --- | ---: |
| Baseline, reference BLAS | 213843200 |
| ERF only, reference BLAS | 213842680 |
| BF16 only, reference BLAS | 213862280 |

## Actual operator comparisons

The independent operator matrix passed. It calls real `_C` GELU/SiLU and fused-MoE operators from each complete library,
in separate processes, using existing repository tolerances for GELU and exact
BF16 bits for the rounding reproducer. Its command is:

```sh
LD_PRELOAD=/usr/lib/llvm-18/lib/libomp.so \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.venv/bin/python check-public-op-matrix.py
```

| Actual operator | Baseline | ERF fix only | BF16 fix only |
| --- | --- | --- | --- |
| Exact GELU, FP32 | 10 NaNs | Finite, matches Torch | 10 NaNs |
| Exact GELU, BF16 | 10 NaNs | Finite, matches Torch | 10 NaNs |
| Tanh GELU, FP32/BF16 | Matches Torch | Matches Torch | Matches Torch |
| Fused MoE exact GELU, BF16 | 512 NaNs | 512 finite outputs, exactly equal to reference | 512 NaNs |
| SiLU(0.125), BF16 | `0x3d87`, wrong | `0x3d87`, wrong | `0x3d88`, exact reference match |

The actual outputs and child-process logs are in `evidence/`. The matrix asserts
both fixed behavior and persistence of the other independent bug. These are
complete-extension operator comparisons, not a run of the repository pytest
suite or a model/serving test. Only VLEN 256 with agnostic ones was used here;
the earlier isolated source-method probes also covered VLEN 128 and other policies.

Full repository pytest, real-weight `LLM` validation, native RVV hardware,
upstream CI and human submission requirements remain separate gates. A pinned
public Pythia-14m model has been downloaded; no model result is claimed here.
Dense CPU GELU uses Torch's native path, so its model smoke test alone would
not establish coverage of the ERF fix.
