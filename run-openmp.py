"""Run the real CPU-platform regression on isolated base and fixed sources."""

import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

repo_arg, base, fixed = sys.argv[1:]
repo = Path(repo_arg).resolve()
output = Path("results/openmp").resolve()
output.mkdir(parents=True, exist_ok=True)
test_path = "tests/platforms/test_cpu.py"
test_source = subprocess.check_output(
    ["git", "-C", str(repo), "show", f"{fixed}:{test_path}"]
)
command = [
    sys.executable,
    "-m",
    "pytest",
    "--noconftest",
    "-q",
    test_path,
    "-k",
    "riscv_preloads_bundled_openmp",
]
results = []
with tempfile.TemporaryDirectory(prefix="vllm-openmp-") as temporary:
    for label, ref in (("baseline", base), ("fixed", fixed)):
        checkout = Path(temporary) / label
        checkout.mkdir()
        archive = subprocess.check_output(
            ["git", "-C", str(repo), "archive", ref, "vllm"]
        )
        with tarfile.open(fileobj=io.BytesIO(archive)) as source:
            source.extractall(checkout, filter="data")
        target = checkout / test_path
        target.parent.mkdir(parents=True)
        target.write_bytes(test_source)
        environment = os.environ | {
            "VLLM_TARGET_DEVICE": "cpu",
            "VLLM_PLUGINS": "",
            "PYTHONPATH": str(checkout),
        }
        run = subprocess.run(
            command,
            check=False,
            cwd=checkout,
            env=environment,
            capture_output=True,
            text=True,
        )
        log = run.stdout + run.stderr
        (output / f"{label}.log").write_text(log)
        results.append({"variant": label, "ref": ref, "returncode": run.returncode})
        print(label, "exit=", run.returncode)

(output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
if (
    [result["returncode"] for result in results] != [1, 0]
    or "2 failed, 4 passed" not in (output / "baseline.log").read_text()
    or "6 passed" not in (output / "fixed.log").read_text()
):
    raise SystemExit("Unexpected results; inspect results/openmp/*.log")
