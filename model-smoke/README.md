# Pinned Pythia-14m BF16 model smoke comparison

The real `LLM` entry point and `CPUWorker` completed two prompts with four
generated tokens each on Ubuntu 24.04 RISC-V Linux/QEMU. The repository's default
`check_logprobs_close` passed: seven of eight greedy tokens matched, and one
punctuation mismatch was retained as a warning. This is a smoke/reference
comparison, not an accuracy benchmark or exact numerical-equivalence claim.

| Prompt | Hugging Face reference tokens | vLLM tokens |
| --- | --- | --- |
| `The capital of France is` | 253, 1986, 2077, **15** | 253, 1986, 2077, **13** |
| `A computer can` | 320, 908, 281, 897 | 320, 908, 281, 897 |

For the mismatch, each generated token is in the other's top-5. The existing
helper accepts this with a warning and stops checking that sequence at the first
mismatch. It does not establish a bound on logprob numerical differences.
The complete warning, load messages and outputs remain in `run.txt`.

## Configuration and observed coverage

- Model: `EleutherAI/pythia-14m`, revision
  `cf967c0a9a04383db6f7b1108d86b2962634b4ac`, real public trained weights.
- Both reference and vLLM use BF16. Greedy sampling, seed 0, two prompts,
  maximum sequence length 128, TP=1, `uni` executor, eager mode, block size 32,
  64 MiB KV cache, prefix caching and chunked prefill disabled.
- QEMU 11.1.2, RVV VLEN 256, Zvfh, agnostic lanes filled with ones; Torch
  2.14.0+cpu. Python package at base `ca8dbccf204abfcdf9dcb178f7e108c06b224b0f`;
  independent BF16-only complete extension at
  `174e56854703c0b9ede89e0a41f9e04dc4c59bc1`.
- The actual run selected the Transformers modeling backend. oneDNN linear
  creation failed and the existing Torch linear fallback executed. These
  messages are preserved; no model implementation or dispatch was patched.
- The CPU profiler observed 24 `_C::cpu_attention_with_kv_cache` calls and
  24 `_C::cpu_attn_reshape_and_cache` calls, plus scheduler, slot mapping and
  greedy-argmax operations. For this RISC-V build and head/block size 32, source
  tracing supports RVV attention/software BF16 conversion coverage. The counts
  do not directly instrument the inlined conversion or record an ISA argument.

`uni` executes a real CPUWorker in the main process; it does not prove that the
OpenMP PR's preload is effective in a newly spawned worker. No
`_C::gelu_and_mul` was observed, so this model does not cover the ERF fix.
There is no base/fixed **model** comparison in this record; the numerical bugs'
base/fixed comparisons are in the separate operator and repository-test records.

## Reproduction

Build the complete BF16-only library and install the real packages from
[full-extension](../full-extension/README.md) and
[runtime-dependencies](../runtime-dependencies/README.md). The identical
oneDNN declaration and reference-BLAS adaptations still apply. The additional
`link-runtime-on-guest.py` records linking actual LLVM OpenMP with `-lomp` and
an rpath, allowing model startup without a forced `LD_PRELOAD`.

Export a complete base snapshot into `~/vllm-validation/model-source` and link
its `vllm/_C.abi3.so` to
`~/vllm-validation/build-base/_C-bf16-linked-openmp.abi3.so`. Download the six
allowlisted files in `model-files.json` from their pinned revision URLs into
`~/vllm-validation/pythia-14m` and check their SHA256 values. The original
archive manifest and transfer/checking script are also retained.

From `~/vllm-validation`, run the unchanged captured checker:

```sh
env -u LD_PRELOAD \
  CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13 \
  TORCHINDUCTOR_COMPILE_THREADS=1 TORCHINDUCTOR_CACHE_DIR="$PWD/inductor-cache" \
  VLLM_TARGET_DEVICE=cpu VLLM_PLUGINS= VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  VLLM_CPU_OMP_THREADS_BIND=0 VLLM_NO_USAGE_STATS=1 \
  HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  PYTHONPATH="$PWD/model-source" timeout 60m .venv/bin/python check-real-model.py
```

An earlier attempt reached model initialization but failed because the
unversioned `g++` was absent. `compiler-discovery-failure.txt` preserves that
failure; selecting the actually installed GCC 13 through standard `CC`/`CXX`
resolved it. Assertions, helper defaults and backend source were unchanged.

The completed JSON and log demonstrate that generation, finite-output assertions
and reference comparison finished. The original terminal session's final exit
code was not retained; no recovered exit-zero claim is made.

Native silicon, broader accuracy evaluation, ERF changed-path model coverage,
OpenMP spawned-worker validation, upstream CI and human review/test confirmation
remain separate requirements.
