"""Verify unchanged dependency commands and headers before reusing oneDNN."""

import argparse
import hashlib
import json
import shlex
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tree_hash(root):
    entries = {
        str(p.relative_to(root)): digest(p)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }
    return {
        "files": len(entries),
        "sha256": hashlib.sha256(
            json.dumps(entries, sort_keys=True).encode()
        ).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant")
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()
    validation = Path.home() / "vllm-validation"
    root = validation / "main-integration-build"
    old_build = validation / "build-base"
    new_build = root / "build" / args.variant
    results = root / "results" / args.variant
    old = json.loads((old_build / "compile_commands.json").read_text())
    new = json.loads((new_build / "compile_commands.json").read_text())
    normalized = {}
    for label, records in (("old", old), ("new", new)):
        values = {}
        for entry in records:
            if "/oneDNN-3.13/" not in entry["file"]:
                continue
            command = entry["command"]
            for old_path, new_path in (
                (str(new_build), str(old_build)),
                (str(root / "source" / args.prefix), str(validation / "source")),
                (str(validation / "main-venv"), str(validation / ".venv")),
            ):
                command = command.replace(old_path, new_path)
            values[entry["file"]] = shlex.split(command)
        normalized[label] = values
    changes = [
        name
        for name in set(normalized["old"]) | set(normalized["new"])
        if normalized["old"].get(name) != normalized["new"].get(name)
    ]
    old_headers = tree_hash(old_build / "_deps/onednn-build/include")
    new_headers = tree_hash(new_build / "_deps/onednn-build/include")
    old_torch = tree_hash(
        validation / ".venv/lib/python3.12/site-packages/torch/include"
    )
    new_torch = tree_hash(
        validation / "main-venv/lib/python3.12/site-packages/torch/include"
    )
    library = old_build / "_deps/onednn-build/src/libdnnl.a"
    library_sha = digest(library)
    expected_sha = "7d2a8be27c186a7f1604e25a6011822e090f695c49b50598aef60f978d5024b3"
    report = {
        "variant": args.variant,
        "old_units": len(normalized["old"]),
        "new_units": len(normalized["new"]),
        "changed_commands": changes,
        "old_generated_headers": old_headers,
        "new_generated_headers": new_headers,
        "old_torch_include": old_torch,
        "new_torch_include": new_torch,
        "library": str(library),
        "library_sha256": library_sha,
        "expected_prior_library_sha256": expected_sha,
        "normalization": "Only relocated build/source/main-venv path prefixes",
    }
    report["reuse_permitted"] = bool(
        len(normalized["new"]) == 210
        and not changes
        and old_headers == new_headers
        and old_headers["files"] > 0
        and old_torch == new_torch
        and old_torch["files"] > 0
        and library_sha == expected_sha
    )
    (results / "dependency-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if not report["reuse_permitted"]:
        (results / "dependency-command-differences.json").write_text(
            json.dumps(normalized, indent=2) + "\n"
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
