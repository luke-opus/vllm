"""Copy selected guest evidence verbatim and produce a host publication manifest."""

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
GUEST = "/home/vllmtest/vllm-validation/main-integration-runtime"
SSH = [
    "ssh",
    "-i",
    str(WORKSPACE / "work/riscv-linux/id_ed25519"),
    "-p",
    "22264",
    "-o",
    "BatchMode=yes",
    "-o",
    "StrictHostKeyChecking=yes",
    "-o",
    "UserKnownHostsFile="
    + str(WORKSPACE / "work/riscv-linux/known_hosts-provisioned-v3"),
    "vllmtest@127.0.0.1",
]

GUEST_EXPORT = r"""
import hashlib, io, json, sys, tarfile
from pathlib import Path
root = Path("/home/vllmtest/vllm-validation/main-integration-runtime")
top = {
    "main_runtime_common.py", "run-main-repository.py", "main_import_probe.py",
    "check-main-model.py", "run-main-model-pair.py", "supervise-main.py",
    "driver-inputs.json", "model-files.json", "prepared-driver-sha256.json",
    "deployment-check.json", "README.txt", "inventory-runtime.py",
    "check-runtime.py", "check-runtime-attempt1.py", "runtime-check-attempt1.txt",
    "old-before.json", "copied-before.json", "old-after.json", "new-after.json",
    "old-before-freeze.txt", "old-after-freeze.txt", "new-after-freeze.txt",
    "old-after-pip-check.txt", "pip-check.txt", "pytest-version.txt",
    "runtime-check.json", "runtime-check.txt", "inode-isolation.json",
    "install-transformers.txt", "guest-wheel-sha256.txt", "wheel-provenance.json",
    "vllm-distribution-metadata.json",
    "resource-monitor.py",
    "resource-monitor-openmp.py",
    "revision-map.json", "tmpdir-preflight-02.json",
}
paths = [root / name for name in top if (root / name).is_file()]
paths.extend(p for p in (root / "inputs").glob("*.py") if p.is_file())
paths.extend(p for p in root.glob("*-supervisor-*.txt") if p.is_file())
paths.extend(p for p in root.glob("resource-*.txt") if p.is_file())
paths.extend(p for p in root.glob("resource-*.json") if p.is_file())
paths.extend(p for p in root.glob("parallel-*.json") if p.is_file())
paths.extend(p for p in root.glob("deployment-check*.json") if p.is_file())
for revision in (root / "driver-revisions").glob("*"):
    paths.extend(p for p in revision.iterdir() if p.is_file() and p.suffix in (".py", ".json", ".txt"))
for run in (root / "runs").glob("*"):
    if not run.is_dir(): continue
    for name in ("repository-results.json", "pair-results.json", "base.json", "fixed.json",
                 "base.txt", "fixed.txt", "comparison-details.json"):
        if (run / name).is_file(): paths.append(run / name)
    for case in run.glob("*"):
        if not case.is_dir() or case.name in ("source", "cache", "tmp", "hf-cache", "inductor-cache"):
            continue
        for name in ("pytest.xml", "pytest.txt", "actual-imports.json"):
            if (case / name).is_file(): paths.append(case / name)
for run in (root / "supervision").glob("*"):
    for name in ("driver.txt", "driver-exit-and-restoration.json"):
        if (run / name).is_file(): paths.append(run / name)
paths = sorted(set(paths))
manifest = []
with tarfile.open(fileobj=sys.stdout.buffer, mode="w|") as archive:
    for path in paths:
        assert not path.is_symlink()
        data = path.read_bytes()
        rel = str(path.relative_to(root))
        info = tarfile.TarInfo(rel)
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
        manifest.append({"path": rel, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    data = (json.dumps({"root": str(root), "files": manifest}, indent=2) + "\n").encode()
    info = tarfile.TarInfo("guest-export-manifest.json")
    info.size = len(data)
    archive.addfile(info, io.BytesIO(data))
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args()
    if not args.collect:
        raise SystemExit("Use --collect only after the active run has completed")
    command = SSH + [
        "PYTHONDONTWRITEBYTECODE=1 /home/vllmtest/vllm-validation/main-venv/bin/python -"
    ]
    result = subprocess.run(
        command,
        input=GUEST_EXPORT.encode(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    with tarfile.open(fileobj=io.BytesIO(result.stdout), mode="r:") as archive:
        for item in archive:
            relative = Path(item.name)
            assert (
                item.isfile()
                and not relative.is_absolute()
                and ".." not in relative.parts
            )
            target = HERE / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.extractfile(item).read())
    guest = json.loads((HERE / "guest-export-manifest.json").read_text())
    selected = {record["path"] for record in guest["files"]}
    for record in guest["files"]:
        data = (HERE / record["path"]).read_bytes()
        assert len(data) == record["bytes"]
        assert hashlib.sha256(data).hexdigest() == record["sha256"]
    selected.update(
        {
            "guest-export-manifest.json",
            "collect-runtime-evidence.py",
            "copy-audit.json",
            "final-package-audit.json",
            "preparation-result.json",
            "transformers-5.22.0-METADATA",
            "transformers-5.22.0-WHEEL",
            "transformers-5.22.0-pypi.json",
        }
    )
    if (HERE / "RESULTS.txt").exists():
        selected.add("RESULTS.txt")
    selected.update(p.name for p in HERE.glob("resource-*.txt") if p.is_file())
    selected.update(p.name for p in HERE.glob("resource-*.json") if p.is_file())
    selected.update(p.name for p in HERE.glob("parallel-*.json") if p.is_file())
    files = []
    for name in sorted(selected):
        data = (HERE / name).read_bytes()
        files.append(
            {
                "path": name,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    manifest = {
        "root": str(HERE),
        "files": files,
        "excluded": [
            "wheel binaries",
            "package/cache directories",
            "full source trees",
            "input model weights and tokenizer files (model-files.json retained)",
            "bytecode",
            "pytest temporary compiled fixtures",
            "build DSOs/objects",
        ],
    }
    (HERE / "publication-files.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "copied_guest_files": len(guest["files"]),
                "publication_files": len(files),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
