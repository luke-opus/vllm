"""Replay a target CMake compilation with host Zig and the target's real headers."""

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("filename", nargs="?", default="/csrc/cpu/activation.cpp")
parser.add_argument("--include-c-math", action="store_true")
parser.add_argument("--math-namespace-compat", action="store_true")
parser.add_argument("--reference-blas", action="store_true")
args = parser.parse_args()
root = Path(__file__).resolve().parent
mirror = root / "vllm-validation"
sysroot = root / "sysroot"
zig = Path("/Users/luke/Documents/Codex/2026-10-10/xian-2/work/cross-tools/ziglang/zig")
database = json.loads((mirror / "build-base/compile_commands.json").read_text())
entries = [entry for entry in database if entry["file"].endswith(args.filename)]
assert len(entries) == 1, len(entries)
entry = entries[0]
original = shlex.split(entry["command"])
mapped = []
for value in original[1:]:
    if value.startswith(("-march=", "-mrvv-vector-bits=")):
        continue
    value = value.replace("/home/vllmtest/vllm-validation", str(mirror))
    if value.startswith("/usr/include"):
        value = str(sysroot) + value
    mapped.append(value)
command = [
    str(zig),
    "c++",
    "-target",
    "riscv64-linux-gnu.2.39",
    "-mcpu=generic_rv64+m+a+f+d+c+v+zicsr+zifencei+zvfh+zvl256b",
    "-mrvv-vector-bits=256",
    "--sysroot",
    str(sysroot),
    "-nostdinc",
    "-nostdinc++",
]
for include in (
    "usr/include/c++/13",
    "usr/include/riscv64-linux-gnu/c++/13",
    "usr/include/c++/13/backward",
    "usr/include/riscv64-linux-gnu",
    "usr/include",
):
    command.extend(["-isystem", str(sysroot / include)])
command.extend(["-isystem", str(zig.parent / "lib/include")])
if args.include_c_math:
    command.extend(["-include", "math.h"])
if args.math_namespace_compat:
    command.extend(["-include", str(root / "compat-math.hpp")])
command.extend(mapped)
if args.reference_blas:
    command.append("-UVLLM_HAS_OPENBLAS")
output = mirror / "build-base" / mapped[mapped.index("-o") + 1]
output.parent.mkdir(parents=True, exist_ok=True)
env = {
    **os.environ,
    "ZIG_GLOBAL_CACHE_DIR": str(root / "zig-cache"),
    "ZIG_LOCAL_CACHE_DIR": str(root / "zig-local-cache"),
}
record = root / "compile-one-command.json"
record.write_text(
    json.dumps({"native_command": original, "cross_command": command}, indent=2) + "\n"
)
print("Cross-compiling", entry["file"], flush=True)
result = subprocess.run(command, cwd=mirror / "build-base", env=env, check=False)
print("Compiler exit:", result.returncode, flush=True)
raise SystemExit(result.returncode)
