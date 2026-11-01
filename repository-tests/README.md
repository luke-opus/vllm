# Actual repository kernel regressions

The selected PR tests ran with the real root `conftest.py`, fixtures, Python
package and independently rebuilt complete CPU extensions on Ubuntu 24.04
RISC-V Linux/QEMU (VLEN 256, agnostic lanes filled with ones).

| Selected test | Base | Independent fix |
| --- | --- | --- |
| `test_cpu_gelu_and_mul` | 2 failed, 2 passed | 4 passed |
| `rounds_to_nearest_even` | 1 failed | 1 passed |

The ERF failures are NaN comparisons in exact GELU; the BF16 failure reports
128/128 different values. The full output and process return codes are retained
here. These are selected regressions, not the entire CPU suite: `-k` deselected
56 and 58 other cases respectively. No conftest bypass, fake objects, skipped
assertions or relaxed tolerances were used in these runs.

Base: `ca8dbccf204abfcdf9dcb178f7e108c06b224b0f`.
ERF test/fix: `7f1353f1a5854545163d49f764552a57f836b2b2`.
BF16 test/fix: `174e56854703c0b9ede89e0a41f9e04dc4c59bc1`.
Each PR test is run unchanged against both libraries. Python source is the
base snapshot, apart from selecting that PR's activation test. The independent
library builds and identical environment adaptations are documented in
[full-extension](../full-extension/README.md); the required real packages and
earlier collection failures are in [runtime-dependencies](../runtime-dependencies/README.md).

## Reproduction

Prepare the target environment and complete libraries as described in those
directories. The retained `run-repository-tests.py` is the original execution
driver: it expects `~/vllm-validation/source`, the three libraries under
`~/vllm-validation/build-base`, and a host HTTP server accessible from QEMU at
`10.0.2.2:8765`. In a disposable host payload directory, export these exact inputs:

```sh
git -C /path/to/vllm show 7f1353f1a5854545163d49f764552a57f836b2b2:tests/kernels/core/test_cpu_activation.py > test_cpu_activation-erf.py
git -C /path/to/vllm show 174e56854703c0b9ede89e0a41f9e04dc4c59bc1:tests/kernels/core/test_cpu_activation.py > test_cpu_activation-bf16.py
/path/to/.venv/bin/python -m http.server 8765 --bind 127.0.0.1
```

The exports used here are checked against their Git blobs in `inputs.json`.
Copy the execution driver to the guest and run it with the guest's virtual
environment. It executes these commands once with each corresponding library:

```sh
.venv/bin/python -m pytest -v source/tests/kernels/core/test_cpu_activation.py -k test_cpu_gelu_and_mul
.venv/bin/python -m pytest -v source/tests/kernels/core/test_cpu_activation.py -k rounds_to_nearest_even
```

The driver sets `PYTHONPATH`, CPU device selection, LLVM OpenMP preload and
one thread, as recorded in its source. Use a disposable source snapshot: it
changes the activation test and extension symlink. Existing development-version
and pytest warnings remain visible in the logs; no version was fabricated.

This is emulated target execution, not native silicon, upstream CI, a full
model evaluation, or the submitting human's review/test confirmation.
