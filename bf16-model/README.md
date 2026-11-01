# BF16 base/fixed comparison with real Pythia-14m weights

Two fresh `LLM`/`CPUWorker` runs completed on Ubuntu 24.04 RISC-V Linux/QEMU,
using independently built base and BF16-fixed complete extensions. Both child
processes and the driver exited **0**, with no timeout. This records an observed
output change, not numerical equivalence or an accuracy improvement.

| Observation | Result |
| --- | --- |
| Greedy tokens | 7 of 8 match; one period changes to a comma |
| Returned top-5 sets | 1 of 8 changes |
| Common returned logprobs at identical prefixes | 39 values |
| Maximum / mean absolute logprob difference | 0.3876380920 / 0.0811952368 |
| Actual affinity / Torch threads, both runs | `[0,1]` / 1 |
| `LD_PRELOAD` / mapped OpenMP runtimes, both runs | Empty / GNU and LLVM |
| Attention / cache profiler calls, each run | 24 / 24 |

`The capital of France is` produced `[253,1986,2077,15]` on base and
`[253,1986,2077,13]` on fixed; `A computer can` produced `[320,908,281,897]`
in both. The returned text and every returned top-5/chosen-token logprob are in
[base.json](base.json) and [fixed.json](fixed.json). These are paired base/fixed
runs, separate from the earlier [HF comparison](../model-smoke/README.md).
They do not capture full-vocabulary distributions.

## Inputs and observed execution

- Base C++ and constant Python source:
  `ca8dbccf204abfcdf9dcb178f7e108c06b224b0f`.
- BF16-only fixed C++: `174e56854703c0b9ede89e0a41f9e04dc4c59bc1`.
  Only the complete extension selected by the source-tree symlink changes
  between runs; Python `cpu.py`, weights, configuration and environment stay
  constant. Both builds retain the same documented oneDNN declaration and
  reference-BLAS adaptations and LLVM OpenMP linkage.
- Real trained `EleutherAI/pythia-14m`, revision
  `cf967c0a9a04383db6f7b1108d86b2962634b4ac`. The driver checks all six pinned
  model-file SHA256 values against [model-files.json](model-files.json).
- BF16, greedy sampling, seed 0, TP=1, eager mode, `uni` executor, maximum
  sequence length 128, block size 32, two sequences, 64 MiB KV cache, prefix
  caching and chunked prefill disabled. `uni` uses a real CPUWorker in each
  checker process; this is not a spawned-worker OpenMP comparison.
- Both runs select the Transformers modeling backend and use the existing
  Torch linear fallback after oneDNN linear initialization fails. The complete
  messages remain in [base.txt](base.txt) and [fixed.txt](fixed.txt).

The profiler counts establish actual attention/cache operator calls.
[coverage.txt](coverage.txt) traces the corresponding RVV attention and software
BF16 conversion source path. The runtime ISA argument and inlined
`float_to_bf16` calls were **not directly instrumented**. This model does not
cover the ERF changed path.

## Provenance and evidence boundaries

[comparison.json](comparison.json) records the link command, shared oneDNN
hash, both library hashes, equal dynamic dependencies/rpaths, environment,
actual child exits and restoration. [driver-exit.json](driver-exit.json) retains
the observed outer command/driver exit. [object-provenance.json](object-provenance.json)
and [object-differences.json](object-differences.json) identify the two 19-object
archives and their relationship to the previous complete-build manifests.

[independent-audit.json](independent-audit.json) is a **separate post-run
inspection**, not assertions added retroactively to the executed driver. It
checks the archive hashes, the reused fixed DSO against the prior
[OpenMP-worker record](../openmp-worker/openmp-worker-run-results.json), the
base `cpu.py` hash, and actual restored symlink/source bytes. The fixed DSO was
captured in live process maps; **base DSO live maps were not captured before its
process exited**. [maps-observations.jsonl](maps-observations.jsonl) preserves
the available observations.

The original driver and checker are unchanged; their guest/local hashes are
in [executed-script-sha256.txt](executed-script-sha256.txt). The driver's unused
timeout branch assigns synthetic status `124` after killing a child; it would
not prove an actual child exit of 124. No timeout occurred in these runs.

## Reproduction

Use an idle, disposable guest prepared through the
[complete-extension build](../full-extension/README.md),
[runtime dependencies](../runtime-dependencies/README.md), and
[model setup](../model-smoke/README.md#reproduction). Under `~/vllm-validation`,
retain the real `.venv`, base `model-source`, pinned `pythia-14m` files, complete
oneDNN/build dependencies, and both `cross-{base,bf16}-reference-blas-objects.tar.gz`
archives. Their recorded hashes are linked from [object-provenance.json](object-provenance.json)
to the [base](../full-extension/evidence/cross-base-reference-blas-manifest.json)
and [BF16](../full-extension/evidence/cross-bf16-reference-blas-manifest.json)
manifests; binaries are not distributed in this evidence directory.

The fixed objects must occupy `build-base/CMakeFiles/`, with
`build-base/_C-bf16-linked-openmp.abi3.so` prepared by the existing
[runtime-link script](../model-smoke/link-runtime-on-guest.py). That preparation
script reads `native-link-commands.json` from the guest's host payload service
at `http://10.0.2.2:8765/`; serve the recorded file there during preparation.
The comparison driver itself uses the local command file. Link
`model-source/vllm/_C.abi3.so` to the fixed DSO initially. Recorded link commands
contain the original guest paths; a differently located build needs its own
CMake-generated native link commands and separately recorded artifact hashes.

Export base `vllm/platforms/cpu.py` as `cpu-platform-openmp-base.py` into the guest
root. Its SHA256 is
`bfa74a5c08f4186b57a3bb83cf1c08fbe1bf35b5522c02020fc1d602dca12f50`.
Copy `run-comparison.py`, `check-model.py`, `native-link-commands.json` and
`model-files.json` into **`~/vllm-validation/bf16-model-comparison/`**. Keep that
execution directory name even though this public evidence folder is named
`bf16-model`. With GCC/G++ 13 and LLVM OpenMP installed as recorded, run:

```sh
cd "$HOME/vllm-validation"
.venv/bin/python -u bf16-model-comparison/run-comparison.py
```

The driver links the base DSO, checks matching runtime dependencies, runs each
checker with a 20-minute limit, and restores the original source/symlink in
`finally`. For descriptive statistics, copy `analyze-results.py` beside the
result JSON files and run:

```sh
.venv/bin/python bf16-model-comparison/analyze-results.py
```

`audit-restoration.py` is the separate captured-run verifier: its fixed expected
hashes and required external live-maps record make it specific to these
artifacts, not a generic new-run driver check. The 27 original allowlisted
files are preserved byte for byte; [artifact-sha256.json](artifact-sha256.json)
covers 25 payload files and [publish-files.txt](publish-files.txt) lists them.
This added README is outside that original manifest.

Physical RVV hardware, broader model accuracy, performance, complete serving,
ERF model coverage, upstream CI and human review/test confirmation remain
separate requirements.
