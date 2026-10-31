"""Keep strict checks while using the verified official NumPy inventory locally."""

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("checkout", type=Path)
parser.add_argument("site", type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parent
cache = root / "numpy-official-objects.inv"
data = cache.read_bytes()
metadata = json.loads((root / "numpy-github-inventory.json").read_text())
assert (
    hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == metadata["sha"]
)
repo = args.checkout.resolve()
original = (repo / "mkdocs.yaml").read_text()
lines = original.splitlines()
matches = [
    i
    for i, line in enumerate(lines)
    if line.strip() == "- https://numpy.org/doc/stable/objects.inv"
]
assert len(matches) == 1
index = matches[0]
indent = lines[index][: len(lines[index]) - len(lines[index].lstrip())]
lines[index] = (
    f"{indent}- url: {cache.as_uri()}\n{indent}  base_url: https://numpy.org/doc/stable"
)
with tempfile.NamedTemporaryFile(
    mode="w", prefix=".mkdocs-inventory-", suffix=".yaml", dir=repo, delete=False
) as handle:
    handle.write("\n".join(lines) + "\n")
    config = Path(handle.name)
try:
    print(
        f"Strict build using official NumPy inventory blob {metadata['sha']}; original base URL and all other settings preserved",
        flush=True,
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mkdocs",
            "build",
            "--strict",
            "--config-file",
            str(config),
            "--site-dir",
            str(args.site.resolve()),
        ],
        cwd=repo,
        check=False,
    )
finally:
    config.unlink()
    assert (repo / "mkdocs.yaml").read_text() == original
raise SystemExit(result.returncode)
