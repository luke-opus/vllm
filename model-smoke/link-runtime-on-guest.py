"""Link Clang's actual OpenMP runtime so model checks need no forced preload."""

import json
import subprocess
import tarfile
from pathlib import Path
from urllib.request import urlopen

root = Path.home() / "vllm-validation"
with tarfile.open(root / "cross-bf16-reference-blas-objects.tar.gz") as bundle:
    for member in bundle:
        assert member.isfile() and member.name.startswith("build-base/CMakeFiles/")
        assert (root / member.name).read_bytes() == bundle.extractfile(member).read()
commands = json.loads(
    urlopen("http://10.0.2.2:8765/native-link-commands.json", timeout=30).read()
)
original = commands[-1]
assert original.endswith(" && :")
command = original.replace(" -o _C.abi3.so ", " -o _C-bf16-linked-openmp.abi3.so ", 1)
command = command.removesuffix(" && :") + (
    " -L/usr/lib/llvm-18/lib -Wl,-rpath,/usr/lib/llvm-18/lib -lomp && :"
)
print("Linking real LLVM OpenMP dependency for subsequent runtime checks", flush=True)
subprocess.run(["/bin/sh", "-c", command], cwd=root / "build-base", check=True)
subprocess.run(
    ["readelf", "-d", "_C-bf16-linked-openmp.abi3.so"],
    cwd=root / "build-base",
    check=True,
)
