"""Use the independent source snapshot after its model process exits."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

root = Path.home() / "vllm-validation"
source = root / "model-source"
model_proc = Path("/proc/13975")
deadline = time.monotonic() + 3600
print("Waiting for the independent model source to become idle", flush=True)
while model_proc.exists():
    assert model_proc.stat().st_uid == os.getuid()
    args = (model_proc / "cmdline").read_bytes().split(b"\0")
    if b"check-real-model.py" not in args:
        break
    assert time.monotonic() < deadline, "Model process still active after wait deadline"
    time.sleep(1)
base = "http://10.0.2.2:8765/"
probe = urlopen(base + "check-native-openmp-config.py", timeout=30).read()
(root / "check-native-openmp-config.py").write_bytes(probe)
original = urlopen(base + "cpu-platform-openmp-base.py", timeout=30).read()
fixed = urlopen(base + "cpu-platform-openmp-fixed.py", timeout=30).read()
path = source / "vllm/platforms/cpu.py"
assert path.read_bytes() == original
env = {
    **os.environ,
    "VLLM_TARGET_DEVICE": "cpu",
    "VLLM_PLUGINS": "",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PYTHONPATH": str(source),
}
env.pop("LD_PRELOAD", None)
records = []
try:
    for variant, content in (("base", original), ("fixed", fixed)):
        path.write_bytes(content)
        log_path = root / f"native-openmp-config-{variant}.txt"
        print(f"Running real Linux/RISC-V OpenMP configuration: {variant}", flush=True)
        with log_path.open("w") as log:
            result = subprocess.run(
                [sys.executable, str(root / "check-native-openmp-config.py"), variant],
                cwd=root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=1200,
                check=False,
            )
        records.append(
            {"variant": variant, "returncode": result.returncode, "log": log_path.name}
        )
        (root / "native-openmp-run-results.json").write_text(
            json.dumps(records, indent=2) + "\n"
        )
        assert result.returncode == 0, records[-1]
        print(log_path.read_text()[-2500:], flush=True)
finally:
    path.write_bytes(original)
print("Actual RISC-V configuration and child-loader checks passed", flush=True)
