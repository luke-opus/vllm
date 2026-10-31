"""Validate and install the real required-version RISC-V llguidance extension."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from urllib.request import urlopen

root = Path.home() / "vllm-validation"
base = "http://10.0.2.2:8765/"
manifest = json.loads(
    urlopen(base + "llguidance-wheel-manifest.json", timeout=30).read()
)
name = manifest["filename"]
assert name.startswith("llguidance-1.7.6-") and name.endswith("_riscv64.whl")
data = urlopen(base + name, timeout=120).read()
assert hashlib.sha256(data).hexdigest() == manifest["sha256"]
wheel = root / name
wheel.write_bytes(data)
subprocess.run(
    [
        str(root / "uv"),
        "pip",
        "install",
        "--python",
        sys.executable,
        "--no-deps",
        str(wheel),
    ],
    cwd=root,
    check=True,
)
subprocess.run(
    [str(root / "uv"), "pip", "check", "--python", sys.executable], cwd=root, check=True
)
script = """
import importlib.metadata
import json
import llguidance
import llguidance._lib as native
assert importlib.metadata.version("llguidance") == "1.7.6"
assert native.__file__.endswith(".so")
grammar = json.loads(llguidance.LLMatcher.grammar_from_regex("a+"))
assert grammar
print(json.dumps({"version": llguidance.get_version(),
                  "native_module": native.__file__, "grammar": grammar}))
"""
result = subprocess.run(
    [sys.executable, "-c", script],
    cwd=root,
    check=True,
    capture_output=True,
    text=True,
    timeout=900,
)
(root / "llguidance-native-validation.json").write_text(result.stdout)
(root / "llguidance-wheel-manifest.json").write_text(
    json.dumps(manifest, indent=2) + "\n"
)
print(result.stdout, flush=True)
print("Real llguidance 1.7.6 native validation passed", flush=True)
