"""Replay a fresh target compilation database with real target headers."""

import argparse
import concurrent.futures
import hashlib
import json
import os
import shlex
import subprocess
import threading
import time
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--production-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    root = args.config.resolve().parent
    mirror = Path(config["mirror"])
    build = mirror / "build" / config["variant"]
    sysroot = Path(config["sysroot"])
    zig = Path(config["zig"])
    records_path = root / f"compile-{config['variant']}.json"
    logs = root / f"logs-{config['variant']}"
    logs.mkdir(exist_ok=True)
    database = json.loads((build / "compile_commands.json").read_text())
    guest_source = Path(config["guest_root"]) / "source" / config["source_prefix"]
    if args.production_only:
        database = [e for e in database if Path(e["file"]).is_relative_to(guest_source)]
        assert len(database) == 21
    mappings = sorted(
        config["path_mappings"].items(), key=lambda item: len(item[0]), reverse=True
    )

    def mapped(value):
        for old, new in mappings:
            value = value.replace(old, new)
        if value.startswith("/usr/include"):
            value = str(sysroot) + value
        return value

    env = {
        **os.environ,
        "ZIG_GLOBAL_CACHE_DIR": str(root / "zig-cache"),
        "ZIG_LOCAL_CACHE_DIR": str(root / "zig-local-cache"),
    }
    lock = threading.Lock()
    failed = threading.Event()
    records = []

    def compile_one(pair):
        index, entry = pair
        if failed.is_set():
            return
        original = entry.get("arguments") or shlex.split(entry["command"])
        values = [
            mapped(v)
            for v in original[1:]
            if not v.startswith(("-march=", "-mrvv-vector-bits="))
        ]
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
        command.extend(["-isystem", str(zig.parent / "lib/include"), *values])
        adaptations = []
        if entry["file"].endswith("/rvv_softmax.cpp"):
            command.extend(["-include", str(root / "compat-math.hpp")])
            adaptations.append("standard declaration only: using std::isfinite")
        if entry["file"].endswith("/sgl-kernels/fla.cpp"):
            command.append("-UVLLM_HAS_OPENBLAS")
            adaptations.append("existing real PyTorch reference BLAS fallback")
        output = build / values[values.index("-o") + 1]
        output.parent.mkdir(parents=True, exist_ok=True)
        source_file = Path(mapped(entry["file"]))
        logfile = logs / f"{index:03d}.txt"
        started = time.monotonic()
        timed_out = False
        with logfile.open("w") as log:
            proc = subprocess.Popen(
                command, cwd=build, env=env, stdout=log, stderr=subprocess.STDOUT
            )
            try:
                returncode = proc.wait(timeout=600)
            except subprocess.TimeoutExpired:
                timed_out = True
                proc.kill()
                returncode = proc.wait()
        record = {
            "index": index,
            "file": entry["file"],
            "source_sha256": digest(source_file),
            "native_command": original,
            "cross_command": command,
            "adaptations": adaptations,
            "returncode": returncode,
            "timed_out": timed_out,
            "seconds": round(time.monotonic() - started, 3),
            "object": str(output.relative_to(build)),
            "log": str(logfile),
        }
        if returncode == 0:
            with output.open("rb") as stream:
                header = stream.read(20)
            assert (
                header[:6] == b"\x7fELF\x02\x01"
                and int.from_bytes(header[18:20], "little") == 243
            )
            record["object_sha256"] = digest(output)
            record["object_bytes"] = output.stat().st_size
        else:
            failed.set()
        with lock:
            records.append(record)
            records_path.write_text(
                json.dumps(
                    {
                        "config": config,
                        "expected_units": len(database),
                        "records": records,
                    },
                    indent=2,
                )
                + "\n"
            )
            print(
                f"{config['variant']} {len(records)}/{len(database)} exit={returncode} {source_file.name}",
                flush=True,
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(compile_one, enumerate(database)))
    assert len(records) == len(database) and not failed.is_set(), (
        "Compilation failed or incomplete"
    )
    print(f"All {len(records)} {config['variant']} units compiled", flush=True)


if __name__ == "__main__":
    main()
