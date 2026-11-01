"""Run actual PR pytest regressions with independently rebuilt complete libraries."""

import json
import os
import subprocess
import sys
from pathlib import Path
from urllib.request import urlopen

root = Path.home() / "vllm-validation"
source = root / "source"
test_path = source / "tests/kernels/core/test_cpu_activation.py"
library_path = source / "vllm/_C.abi3.so"
env = {
    **os.environ,
    "VLLM_TARGET_DEVICE": "cpu",
    "VLLM_PLUGINS": "",
    "LD_PRELOAD": "/usr/lib/llvm-18/lib/libomp.so",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PYTHONPATH": str(source),
}
results = []
for fix, selector, failures in (
    ("erf", "test_cpu_gelu_and_mul", 2),
    ("bf16", "rounds_to_nearest_even", 1),
):
    test_path.write_bytes(
        urlopen(f"http://10.0.2.2:8765/test_cpu_activation-{fix}.py", timeout=30).read()
    )
    for variant in ("base", fix):
        if library_path.is_symlink():
            library_path.unlink()
        assert not library_path.exists()
        library = root / f"build-base/_C-{variant}-reference-blas.abi3.so"
        library_path.symlink_to(library)
        command = [sys.executable, "-m", "pytest", "-v", str(test_path), "-k", selector]
        log_path = root / f"repository-pytest-{fix}-{variant}.txt"
        print(f"Running actual repository pytest: {fix} / {variant}", flush=True)
        with log_path.open("w") as log:
            result = subprocess.run(
                command,
                cwd=source,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=1800,
                check=False,
            )
        output = log_path.read_text()
        record = {
            "fix": fix,
            "variant": variant,
            "command": command,
            "returncode": result.returncode,
            "log": log_path.name,
        }
        results.append(record)
        (root / "repository-pytest-results.json").write_text(
            json.dumps(results, indent=2) + "\n"
        )
        if variant == "base":
            assert result.returncode == 1, record
            assert f"{failures} failed" in output and "AssertionError" in output, record
        else:
            assert result.returncode == 0 and " passed" in output, record
        print(output[-2000:], flush=True)
print("Repository public-op base/fixed pytest regressions passed", flush=True)
