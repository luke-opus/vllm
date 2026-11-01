# RISC-V Linux OpenMP configuration and child loading

Both configurations ran on actual emulated RISC-V Linux, with the real
`VllmConfig`, `DeviceConfig`, CPU detection, affinity, Torch installation and
library files. No OS, architecture or filesystem inputs were mocked.

| Observation | Base | PR #64080 |
| --- | --- | --- |
| Configuration-produced `LD_PRELOAD` | Empty | Torch's real `torch/lib/libgomp.so.1` |
| Fresh child inherited preload | Empty | Same bundled library |
| Child Torch threads | 1 | 1 |
| Real SiLU BF16 result | `0x3d88` | `0x3d88` |

Both children map GOMP and LLVM OpenMP. This proves the configuration change,
child inheritance and successful operator loading. It does not prove runtime
conflict elimination, throughput improvement or complete vLLM worker behavior.

The base Python source is `ca8dbccf204abfcdf9dcb178f7e108c06b224b0f`;
only `vllm/platforms/cpu.py` changes to the file from
`018ef906c8cfc72cdfa21eafac157258c63373cb`. Both variants use the same BF16-fixed
complete extension, linked to LLVM OpenMP through `DT_NEEDED`; this keeps the
parent's preload empty so the policy under test can run. That numerical fix is
held constant and is not part of the OpenMP PR.

## Reproduction

Use the [complete-extension environment](../full-extension/README.md) and
[real runtime dependencies](../runtime-dependencies/README.md), then link the
real LLVM runtime using [the captured link step](../model-smoke/link-runtime-on-guest.py).
Export the two
platform files from the exact refs:

```sh
git -C /path/to/vllm show ca8dbccf204abfcdf9dcb178f7e108c06b224b0f:vllm/platforms/cpu.py > cpu-platform-openmp-base.py
git -C /path/to/vllm show 018ef906c8cfc72cdfa21eafac157258c63373cb:vllm/platforms/cpu.py > cpu-platform-openmp-fixed.py
```

`inputs.json` records the Git blob identities and SHA256 checks of the actual
exports. In an idle, disposable base snapshot named
`~/vllm-validation/model-source`, select each platform file in turn and run:

```sh
cd "$HOME/vllm-validation"
env -u LD_PRELOAD VLLM_TARGET_DEVICE=cpu VLLM_PLUGINS= \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH="$PWD/model-source" \
  .venv/bin/python check-native-openmp-config.py base
```

Use `fixed` for the second invocation and restore the base file afterward.
The actual library name/path is in the checker. `run-native-openmp-config.py`
is retained as the historical execution coordinator, not a generic launcher:
its HTTP address and `/proc/13975` wait were specific to the original session.
It restored the source file in `finally`. Per-process logs and return codes
are included. No native-silicon, full-worker or performance result is implied.
