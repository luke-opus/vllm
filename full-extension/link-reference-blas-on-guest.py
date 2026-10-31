"""Link complete libraries with the same real PyTorch BLAS fallback enabled."""

import hashlib
import json
import subprocess
import tarfile
from pathlib import Path
from urllib.request import urlopen

root = Path.home() / "vllm-validation"
base = "http://10.0.2.2:8765/"
commands = json.loads(urlopen(base + "native-link-commands.json", timeout=30).read())
original = commands[-1]
assert " -o _C.abi3.so " in original
for variant in ("base", "erf", "bf16"):
    name = f"cross-{variant}-reference-blas"
    manifest = json.loads(urlopen(base + name + "-manifest.json", timeout=30).read())
    assert manifest["objects"] == 19 and manifest["variant"] == variant
    archive = root / (name + "-objects.tar.gz")
    with urlopen(base + archive.name, timeout=120) as response:
        archive.write_bytes(response.read())
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == manifest["sha256"]
    with tarfile.open(archive) as bundle:
        members = bundle.getmembers()
        assert len(members) == 19
        assert all(
            m.isfile()
            and m.name.startswith("build-base/CMakeFiles/")
            and m.name.endswith(".o")
            for m in members
        )
        bundle.extractall(root, filter="data")
    output_name = f"_C-{variant}-reference-blas.abi3.so"
    command = original.replace(" -o _C.abi3.so ", f" -o {output_name} ", 1)
    print(
        f"Native linking {variant}, {manifest['source_ref']}, reference BLAS",
        flush=True,
    )
    subprocess.run(["/bin/sh", "-c", command], cwd=root / "build-base", check=True)
    output = root / "build-base" / output_name
    print(f"Linked {output_name}: {output.stat().st_size} bytes", flush=True)
