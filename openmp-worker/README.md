# OpenMP preload through a real spawned CPUWorker

Both the base and PR #64080 platform configurations completed real `LLM`
generation on Ubuntu 24.04 RISC-V Linux/QEMU using a separately spawned
`vllm.v1.worker.cpu_worker.CPUWorker`. Each generated four tokens for each of
two prompts. Tokens, text, all 40 returned token logprobs, cumulative logprobs,
and the remaining recorded output fields are exactly equal between runs.
Both checker processes exited 0 and workers shut down gracefully.

This is a passing base/fixed comparison of preload behavior and model output
preservation. It does not reproduce a baseline model failure or establish a
throughput improvement.

| Observation | Base | PR #64080 platform file |
| --- | --- | --- |
| Parent / worker PID | 1175 / 1353 | 1668 / 1682 |
| Worker class / device | Real CPUWorker / CPU | Real CPUWorker / CPU |
| Initial and current worker `LD_PRELOAD` | Empty | Torch's bundled `torch/lib/libgomp.so.1` |
| `GOMP_CPU_AFFINITY` | Empty | `0` |
| `OMP_PLACES` / `OMP_PROC_BIND` | `{0}` / `true` | Empty / empty |
| Torch threads / `OMP_NUM_THREADS` | 1 / 1 | 1 / 1 |
| Observed CPU affinity | `[0]` | `[0]` |
| Mapped OpenMP libraries | GOMP and LLVM OpenMP | GOMP and LLVM OpenMP |
| Checker return code / timeout | 0 / false | 0 / false |

`LLM.collective_rpc` runs the observer in the real worker after generation.
It reads the worker's actual PID, environment, `/proc/self/environ`,
`/proc/self/maps`, Torch thread count and scheduler affinity. In particular,
the fixed run's `/proc/self/environ` contains the bundled preload at process
startup; this is more than a later Python-environment assignment. No platform
or worker objects, loader paths, or affinity results are mocked.

## Exact inputs and configuration

- Complete base Python snapshot:
  `ca8dbccf204abfcdf9dcb178f7e108c06b224b0f`.
- The only Python source change between runs is `vllm/platforms/cpu.py` from
  PR #64080, `018ef906c8cfc72cdfa21eafac157258c63373cb`.
  [Input identities](../openmp-linux/inputs.json) match both source SHA256
  values recorded in `openmp-worker-run-results.json`.
- Both runs use the same complete BF16-fixed C++ extension from
  `174e56854703c0b9ede89e0a41f9e04dc4c59bc1`, with the previously documented
  identical oneDNN-declaration and reference-BLAS adaptations. The extension
  is linked to actual LLVM OpenMP through `DT_NEEDED`, so the parent can start
  without a forced preload. The BF16 fix is held constant and is not part of
  the OpenMP PR. Recorded library SHA256:
  `a13aadd18a4f7d9ccf275f983e12d425b92965d33fc0b9e39b65029f9af570ad`.
- Real trained model: `EleutherAI/pythia-14m`, revision
  `cf967c0a9a04383db6f7b1108d86b2962634b4ac`; BF16, greedy sampling, seed 0,
  TP=1, `mp` executor with worker method `spawn`, eager mode, block size 32,
  maximum sequence length 128, two sequences, 64 MiB KV cache, prefix caching
  and chunked prefill disabled.
- `VLLM_ENABLE_V1_MULTIPROCESSING=0` keeps the engine core in the parent;
  the separate `mp` executor still spawns the worker, as the distinct PIDs show.
  The driver selects the first CPU in the guest's allowed affinity set and
  uses one OpenMP/OpenBLAS thread. Here that CPU was 0.
- Both runs selected the Transformers modeling backend and fell back from
  oneDNN linear initialization to the existing Torch linear implementation.
  The complete warnings are retained. The base log also preserves a shared
  memory broadcast wait message before generation completed successfully.

The captured outputs for the two prompts are `[253, 1986, 2077, 13]` and
`[320, 908, 281, 897]` in both variants. This comparison is between the base and
fixed OpenMP configurations, not against Hugging Face. The separate
[HF comparison](../model-smoke/README.md) retains its punctuation mismatch and
has a different validation scope.

## Reproduction

Prepare an idle, disposable target workspace at `~/vllm-validation` using the
[complete CPU-extension build](../full-extension/README.md),
[runtime dependencies](../runtime-dependencies/README.md), and
[model setup](../model-smoke/README.md). Keep the real virtual environment at
`.venv`, the verified six model files at `pythia-14m`, and the complete base
Python source at `model-source`. The original build and model setup contain
their own preparation steps; the worker driver below uses local files only.

Follow [the runtime-link preparation](../model-smoke/README.md#reproduction)
and its [recorded linker script](../model-smoke/link-runtime-on-guest.py) to
produce `build-base/_C-bf16-linked-openmp.abi3.so`. Point
`model-source/vllm/_C.abi3.so` to that file. The driver verifies the target and
checks that its SHA256 stays unchanged throughout both runs.

From a checkout containing the exact commits, export these inputs and copy
them into the guest workspace:

```sh
git -C /path/to/vllm show ca8dbccf204abfcdf9dcb178f7e108c06b224b0f:vllm/platforms/cpu.py > cpu-platform-openmp-base.py
git -C /path/to/vllm show 018ef906c8cfc72cdfa21eafac157258c63373cb:vllm/platforms/cpu.py > cpu-platform-openmp-fixed.py
```

Check their hashes against [inputs.json](../openmp-linux/inputs.json). Copy
`check-openmp-worker.py` and `run-openmp-worker.py` from this directory into the
same guest workspace. GCC 13 and G++ 13 must be installed at the paths recorded
in the driver's environment settings. Start with the base platform file in
`model-source/vllm/platforms/cpu.py`, then run in the guest:

```sh
cd "$HOME/vllm-validation"
.venv/bin/python run-openmp-worker.py --root "$PWD"
```

The driver has no HTTP input dependency or hardcoded process ID. It clears
pre-existing preload/affinity variables, selects each local platform file in
turn, runs a fresh checker process with a one-hour timeout, and restores the
original source in `finally`. It verifies that the platform difference is
exactly the RISC-V architecture tuple entry and compares the entire recorded
`outputs` structure with strict equality. This run records `source_restored:
true`, no timeouts, and both return codes 0.

The five original JSON/text outputs and two execution scripts are included
unchanged. This establishes one spawned-worker configuration and output
comparison in emulated Linux. Physical RVV hardware, multiple-thread behavior,
runtime-conflict elimination, throughput, full OpenAI-server behavior, upstream
CI and the submitting human's review/test confirmation remain unverified.
