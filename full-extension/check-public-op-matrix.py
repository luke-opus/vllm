"""Compare independently rebuilt full CPU libraries in isolated child processes."""

import json
import os
import runpy
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
libraries = {
    variant: root / f"build-base/_C-{variant}-reference-blas.abi3.so"
    for variant in ("base", "erf", "bf16")
}
assert os.environ.get("OMP_NUM_THREADS") == "1"
assert os.environ.get("OPENBLAS_NUM_THREADS") == "1"
for variant, library in libraries.items():
    assert library.is_file(), library
    print(f"Running {variant} actual CPU operators", flush=True)
    pid = os.fork()
    if pid == 0:
        with (root / f"public-ops-{variant}-reference-blas-child.log").open("w") as log:
            os.dup2(log.fileno(), 1)
            os.dup2(log.fileno(), 2)
            sys.argv = [
                str(root / "check-public-ops.py"),
                str(library),
                "--output",
                str(root / f"public-ops-{variant}-reference-blas.json"),
            ]
            runpy.run_path(str(root / "check-public-ops.py"), run_name="__main__")
        os._exit(0)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0, (variant, status)
results = {
    variant: json.loads(
        (root / f"public-ops-{variant}-reference-blas.json").read_text()
    )
    for variant in libraries
}
base, erf, bf16 = (results[name] for name in ("base", "erf", "bf16"))
assert base["gelu_torch.float32_none"]["nan_count"] > 0
assert not base["silu_bf16"]["exact_match"]
assert base["silu_bf16"]["output_bits"] == "0x3d87"
for key in (
    "gelu_torch.float32_none",
    "gelu_torch.bfloat16_none",
    "gelu_torch.float32_tanh",
    "gelu_torch.bfloat16_tanh",
):
    assert erf[key]["finite"] and erf[key]["matches_reference"], key
assert erf["moe_gelu_bf16"]["exact_match"]
assert base["moe_gelu_bf16"]["nan_count"] > 0
assert bf16["silu_bf16"]["exact_match"]
assert bf16["silu_bf16"]["output_bits"] == "0x3d88"
assert bf16["gelu_torch.float32_none"]["nan_count"] > 0
(root / "public-op-matrix-reference-blas-results.json").write_text(
    json.dumps(results, indent=2) + "\n"
)
print("Independent ERF and BF16 full CPU operator comparisons passed", flush=True)
