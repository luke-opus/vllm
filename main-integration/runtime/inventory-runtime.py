"""Capture package metadata and key immutable inputs without importing packages."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
prefix = Path(sys.prefix)
site = prefix / "lib/python3.12/site-packages"
files = [prefix / "pyvenv.cfg"]
for relative in (
    "torch/__init__.py", "torch/version.py", "torch/lib/libtorch_cpu.so",
    "torch/lib/libtorch.so", "torch/lib/libc10.so", "torch/lib/libgomp.so.1",
    "transformers/__init__.py", "tokenizers/__init__.py",
    "huggingface_hub/__init__.py", "mistral_common/__init__.py",
):
    path = site / relative
    if path.is_file():
        files.append(path)
files.extend(p for p in (site / "torch/include").rglob("*") if p.is_file())
for directory in site.glob("*.dist-info"):
    for name in ("METADATA", "WHEEL", "RECORD", "direct_url.json"):
        path = directory / name
        if path.is_file():
            files.append(path)
digests = {str(p.relative_to(prefix)): sha(p) for p in sorted(set(files))}
torch_headers = {k: v for k, v in digests.items() if "/torch/include/" in k}
report = {
    "sys_prefix": sys.prefix,
    "sys_base_prefix": sys.base_prefix,
    "executable": sys.executable,
    "python_version": sys.version,
    "python_link": os.readlink(prefix / "bin/python"),
    "packages": sorted(
        [{"name": d.metadata["Name"], "version": d.version}
         for d in importlib.metadata.distributions()],
        key=lambda x: x["name"].lower(),
    ),
    "key_file_sha256": digests,
    "torch_header_count": len(torch_headers),
    "torch_header_manifest_sha256": hashlib.sha256(
        json.dumps(torch_headers, sort_keys=True).encode()
    ).hexdigest(),
}
args.output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: report[k] for k in (
    "sys_prefix", "sys_base_prefix", "executable", "python_version",
    "torch_header_count", "torch_header_manifest_sha256"
)}))
