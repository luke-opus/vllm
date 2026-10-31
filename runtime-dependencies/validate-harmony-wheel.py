"""Validate the actual cross-built Rust extension in an isolated target directory."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from urllib.request import urlopen

root = Path.home() / "vllm-validation"
base = "http://10.0.2.2:8765/"
manifest = json.loads(urlopen(base + "harmony-wheel-manifest.json", timeout=30).read())
name = manifest["filename"]
assert name.startswith("oss_harmony-0.0.12-") and name.endswith("_riscv64.whl")
wheel = root / name
wheel.write_bytes(urlopen(base + name, timeout=120).read())
assert hashlib.sha256(wheel.read_bytes()).hexdigest() == manifest["sha256"]
target = root / "harmony-cross-check"
subprocess.run(
    [
        str(root / "uv"),
        "pip",
        "install",
        "--python",
        sys.executable,
        "--target",
        str(target),
        "--only-binary",
        ":all:",
        "--default-index",
        "https://mirrors.aliyun.com/pypi/simple",
        "--index",
        "https://pypi.riseproject.dev/simple",
        "--index-strategy",
        "unsafe-best-match",
        str(wheel),
    ],
    cwd=root,
    check=True,
)
script = """
import importlib.metadata
import json
import openai_harmony
import openai_harmony.openai_harmony as native
assert importlib.metadata.version("oss-harmony") == "0.0.12"
assert native.__file__.endswith(".so")
encoding = openai_harmony.load_harmony_encoding("HarmonyGptOss")
text = "vLLM RISC-V validation"
tokens = encoding.encode(text)
assert encoding.decode(tokens) == text
print(json.dumps({"version": "0.0.12", "native_module": native.__file__,
                  "round_trip": text, "tokens": tokens}))
"""
result = subprocess.run(
    [sys.executable, "-c", script],
    env={**os.environ, "PYTHONPATH": str(target)},
    check=True,
    capture_output=True,
    text=True,
    timeout=900,
)
print(result.stdout, flush=True)
(root / "harmony-cross-check-result.json").write_text(result.stdout)
(root / "harmony-wheel-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
