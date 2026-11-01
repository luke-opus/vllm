"""Configure an isolated real current-main source snapshot on RISC-V Linux."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import tarfile
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant")
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--extract-only", action="store_true")
    args = parser.parse_args()
    validation = Path.home() / "vllm-validation"
    root = validation / "main-integration-build"
    archive = root / f"main-source-{args.variant}.tar.gz"
    assert digest(archive) == args.sha256
    sources = root / "source"
    source = sources / args.prefix
    if not source.exists():
        sources.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive) as bundle:
            assert all(
                m.name.startswith(args.prefix + "/")
                or (m.name == args.prefix and m.isdir())
                for m in bundle
            )
            bundle.extractall(sources, filter="data")
    if args.extract_only:
        print(f"Verified and extracted {args.variant}", flush=True)
        return
    build = root / "build" / args.variant
    results = root / "results" / args.variant
    results.mkdir(parents=True, exist_ok=True)
    command = [
        "cmake",
        "-S",
        str(source),
        "-B",
        str(build),
        "-G",
        "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_C_COMPILER=/usr/bin/clang-18",
        "-DCMAKE_CXX_COMPILER=/usr/bin/clang++-18",
        "-DVLLM_TARGET_DEVICE=cpu",
        "-DVLLM_RVV_VLEN=256",
        f"-DVLLM_PYTHON_EXECUTABLE={validation}/main-venv/bin/python",
        f"-DCMAKE_INSTALL_PREFIX={source}",
        f"-DFETCHCONTENT_SOURCE_DIR_ONEDNN={validation}/oneDNN-3.13",
    ]
    timed_out = False
    with (results / "configure.txt").open("w") as log:
        proc = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            returncode = proc.wait(timeout=900)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(proc.pid, signal.SIGKILL)
            returncode = proc.wait()
    report = {
        "script_sha256": digest(Path(__file__)),
        "variant": args.variant,
        "archive_sha256": args.sha256,
        "source": str(source),
        "build": str(build),
        "command": command,
        "returncode": returncode,
        "timed_out": timed_out,
    }
    (results / "configure.json").write_text(json.dumps(report, indent=2) + "\n")
    assert returncode == 0, (results / "configure.txt").read_text()[-4000:]
    commands = subprocess.check_output(
        ["ninja", "-C", str(build), "-t", "commands", "_C"], text=True
    )
    (results / "ninja-commands.txt").write_text(commands)
    links = [
        line
        for line in commands.splitlines()
        if " -shared " in line and " -o _C.abi3.so " in line
    ]
    assert len(links) == 1, links
    (results / "link-command.json").write_text(json.dumps(links[0]) + "\n")
    database = json.loads((build / "compile_commands.json").read_text())
    production = [
        entry for entry in database if Path(entry["file"]).is_relative_to(source)
    ]
    report["compilation_units"] = len(database)
    report["production_units"] = len(production)
    report["production_sources"] = [
        str(Path(e["file"]).relative_to(source)) for e in production
    ]
    assert len(production) == 21, report
    generated = source / "csrc/cpu/cpu_attn_dispatch_generated.h"
    report["attention_header_sha256"] = digest(generated)
    (results / "configure.json").write_text(json.dumps(report, indent=2) + "\n")
    bundle_path = root / f"configure-{args.variant}.tar.gz"
    with tarfile.open(bundle_path, "w:gz") as bundle:
        for path in (
            build / "CMakeCache.txt",
            build / "compile_commands.json",
            generated,
        ):
            bundle.add(path, arcname=str(path.relative_to(root)))
        bundle.add(
            build / "_deps/onednn-build/include",
            arcname=f"build/{args.variant}/_deps/onednn-build/include",
        )
        bundle.add(results, arcname=f"results/{args.variant}")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
