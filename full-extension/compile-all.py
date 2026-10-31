"""Replay all target CMake compilation units with a host RISC-V compiler."""

import argparse
import concurrent.futures
import json
import os
import shlex
import subprocess
import threading
import time
from pathlib import Path

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source-only", action="store_true")
parser.add_argument("--variant", default="base")
args = parser.parse_args()
assert args.variant in ("base", "erf", "bf16")
mirror = root / "vllm-validation"
sysroot = root / "sysroot"
zig = Path("/Users/luke/Documents/Codex/2026-10-10/xian-2/work/cross-tools/ziglang/zig")
database = json.loads((mirror / "build-base/compile_commands.json").read_text())
if args.source_only:
    database = [entry for entry in database if "/source/csrc/" in entry["file"]]
logs = root / (
    "compile-logs" if args.variant == "base" else f"compile-logs-{args.variant}"
)
result_path = root / (
    "compile-results.json"
    if args.variant == "base"
    else f"compile-results-{args.variant}.json"
)
logs.mkdir(exist_ok=True)
env = {
    **os.environ,
    "ZIG_GLOBAL_CACHE_DIR": str(root / "zig-cache"),
    "ZIG_LOCAL_CACHE_DIR": str(root / "zig-local-cache"),
}
failed = threading.Event()
lock = threading.Lock()
results = []
next_index = 0


def compile_entry(index, entry):
    original = shlex.split(entry["command"])
    assert "-c" in original and "-o" in original
    mapped = []
    for value in original[1:]:
        if value.startswith(("-march=", "-mrvv-vector-bits=")):
            continue
        value = value.replace("/home/vllmtest/vllm-validation", str(mirror))
        if value.startswith("/usr/include"):
            value = str(sysroot) + value
        mapped.append(value)
    frontend = "c++" if "++" in Path(original[0]).name else "cc"
    command = [
        str(zig),
        frontend,
        "-target",
        "riscv64-linux-gnu.2.39",
        "-mcpu=generic_rv64+m+a+f+d+c+v+zicsr+zifencei+zvfh+zvl256b",
        "-mrvv-vector-bits=256",
        "--sysroot",
        str(sysroot),
        "-nostdinc",
    ]
    if frontend == "c++":
        command.append("-nostdinc++")
        for include in (
            "usr/include/c++/13",
            "usr/include/riscv64-linux-gnu/c++/13",
            "usr/include/c++/13/backward",
        ):
            command.extend(["-isystem", str(sysroot / include)])
    for include in ("usr/include/riscv64-linux-gnu", "usr/include"):
        command.extend(["-isystem", str(sysroot / include)])
    command.extend(["-isystem", str(zig.parent / "lib/include"), *mapped])
    output = mirror / "build-base" / mapped[mapped.index("-o") + 1]
    output.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    logfile = logs / f"{index:03d}.log"
    with logfile.open("w") as handle:
        result = subprocess.run(
            command,
            cwd=mirror / "build-base",
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            timeout=600,
            check=False,
        )
    record = {
        "index": index,
        "file": entry["file"],
        "returncode": result.returncode,
        "seconds": round(time.monotonic() - start, 2),
        "native_command": original,
        "cross_command": command,
        "log": str(logfile),
    }
    with lock:
        results.append(record)
        result_path.write_text(json.dumps(results, indent=2) + "\n")
        print(
            f"[{len(results)}/{len(database)}] exit={result.returncode} {entry['file']} ({record['seconds']}s)",
            flush=True,
        )
    if result.returncode:
        failed.set()


def worker():
    global next_index
    while not failed.is_set():
        with lock:
            if next_index >= len(database):
                return
            index = next_index
            next_index += 1
        try:
            compile_entry(index, database[index])
        except Exception:
            failed.set()
            raise


with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    for future in [pool.submit(worker) for _ in range(2)]:
        future.result()
if failed.is_set() or len(results) != len(database):
    raise SystemExit(1)
print(
    "All target compilation units completed; native target linking remains", flush=True
)
