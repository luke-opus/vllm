"""Verify uploaded real objects and link an isolated complete CPU extension."""

import argparse
import hashlib
import json
import os
import shutil
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
    parser.add_argument("--reuse-onednn", action="store_true")
    args = parser.parse_args()
    root = Path.home() / "vllm-validation/main-integration-build"
    build = root / "build" / args.variant
    results = root / "results" / args.variant
    manifest = json.loads((root / f"objects-{args.variant}.json").read_text())
    archive = root / f"objects-{args.variant}.tar.gz"
    assert digest(archive) == manifest["archive_sha256"]
    assert digest(build / "compile_commands.json") == manifest["database_sha256"]
    with tarfile.open(archive) as bundle:
        members = bundle.getmembers()
        assert {m.name for m in members} == set(manifest["objects"])
        assert all(
            m.isfile() and m.name.endswith(".o") and not Path(m.name).is_absolute()
            for m in members
        )
        bundle.extractall(build, filter="data")
    for name, expected in manifest["objects"].items():
        assert digest(build / name) == expected
    if args.reuse_onednn:
        audit = json.loads((results / "dependency-audit.json").read_text())
        assert audit["reuse_permitted"]
        cached = Path(audit["library"])
        assert digest(cached) == audit["library_sha256"]
        destination = build / "_deps/onednn-build/src/libdnnl.a"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached, destination)
        assert digest(destination) == audit["library_sha256"]
    else:
        commands = (results / "ninja-commands.txt").read_text().splitlines()
        archive_commands = [
            c
            for c in commands
            if "llvm-ar" in c and "libdnnl.a" in c and " -c " not in c
        ]
        assert len(archive_commands) == 1
        subprocess.run(
            ["/bin/sh", "-c", archive_commands[0]], cwd=build, check=True, timeout=600
        )
    original = json.loads((results / "link-command.json").read_text())
    assert original.endswith(" && :")
    command = (
        original.removesuffix(" && :")
        + " -L/usr/lib/llvm-18/lib -Wl,-rpath,/usr/lib/llvm-18/lib -lomp && :"
    )
    timed_out = False
    with (results / "link.txt").open("w") as log:
        proc = subprocess.Popen(
            ["/bin/sh", "-c", command],
            cwd=build,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            returncode = proc.wait(timeout=600)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(proc.pid, signal.SIGKILL)
            returncode = proc.wait()
    report = {
        "script_sha256": digest(Path(__file__)),
        "variant": args.variant,
        "source_identity": manifest["source_identity"],
        "command": command,
        "returncode": returncode,
        "timed_out": timed_out,
        "object_count": len(manifest["objects"]),
        "reused_onednn": args.reuse_onednn,
    }
    (results / "link.json").write_text(json.dumps(report, indent=2) + "\n")
    assert returncode == 0
    library = build / "_C.abi3.so"
    report.update(
        {
            "library": str(library),
            "library_sha256": digest(library),
            "bytes": library.stat().st_size,
        }
    )
    dynamic = subprocess.check_output(["readelf", "-d", str(library)], text=True)
    (results / "dynamic.txt").write_text(dynamic)
    assert "libomp.so.5" in dynamic
    (results / "link.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
