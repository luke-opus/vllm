"""Observe guest RSS/high-water marks and background load without changing a run."""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path


def snapshot():
    mem = {
        line.split(":")[0]: int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
    }
    processes = []
    compilers = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdecimal():
            continue
        try:
            name = (proc / "comm").read_text().strip()
            command = (
                (proc / "cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode(errors="replace")
                .strip()
            )
            if name in (
                "gcc-13",
                "g++-13",
                "gcc",
                "g++",
                "cc1",
                "cc1plus",
                "clang",
                "clang++",
                "ld",
                "ld.lld",
            ):
                compilers.append({"pid": int(proc.name), "name": name})
            if not name.startswith("python") or "/main-venv/bin/python" not in command:
                continue
            if not any(
                token in command
                for token in (
                    "check-main-model.py",
                    "run-main-model-pair.py",
                    "pytest",
                    "run-main-repository.py",
                )
            ):
                continue
            status = (proc / "status").read_text().splitlines()
            values = {
                line.split(":")[0]: int(line.split()[1]) * 1024
                for line in status
                if line.startswith(("VmRSS:", "VmHWM:"))
            }
            processes.append({"pid": int(proc.name), "command": command, **values})
        except (FileNotFoundError, ProcessLookupError):
            continue
    return {
        "utc": datetime.now(timezone.utc).isoformat(),
        "MemAvailable": mem["MemAvailable"],
        "SwapTotal": mem["SwapTotal"],
        "SwapFree": mem["SwapFree"],
        "processes": processes,
        "compilers": compilers,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exit-marker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path("/home/vllmtest/vllm-validation/main-integration-runtime")
    assert args.output.is_relative_to(root) and args.exit_marker.is_relative_to(root)
    report = {
        "scope": "External sampled /proc RSS and VmHWM observations; durations are not performance evidence",
        "interval_seconds": 5,
        "began_after_model_start": True,
        "samples": [],
    }
    while True:
        report["samples"].append(snapshot())
        complete = args.exit_marker.exists()
        report["exit_marker_observed"] = complete
        if complete:
            report["observed_exit_marker"] = args.exit_marker.read_text()
        pending = args.output.with_suffix(".json.tmp")
        pending.write_text(json.dumps(report, indent=2) + "\n")
        pending.replace(args.output)
        if complete:
            return
        time.sleep(5)


if __name__ == "__main__":
    main()
