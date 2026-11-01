"""Recreate the original mixed gzip recipes, retaining the first audit script."""

import argparse
import gzip
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path


def finish(process):
    timed_out = False
    try:
        _, error = process.communicate(timeout=300)
    except subprocess.TimeoutExpired:
        timed_out = True
        process.kill()
        _, error = process.communicate()
    return error, timed_out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    helper_path = here / "recreate-source-archives.py"
    spec = importlib.util.spec_from_file_location("archive_audit", helper_path)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    inputs_path = here / "source-archive-recreation-inputs.json"
    inputs = json.loads(inputs_path.read_text())
    repo = args.repo.resolve()
    output = args.output.resolve()
    reference = args.reference_dir.resolve() if args.reference_dir else None
    if reference == output or (output.exists() and any(output.iterdir())):
        parser.error("Output must be a new/empty directory distinct from the reference")
    git = shutil.which("git")
    compressor = shutil.which("gzip")
    if git is None or compressor is None:
        parser.error("Real git and gzip executables are required")
    for item in inputs["variants"]:
        kind = subprocess.check_output(
            [git, "-C", str(repo), "cat-file", "-t", item["ref"]], text=True
        ).strip()
        if kind != item["object_type"]:
            parser.error(f"Wrong/missing frozen object type for {item['variant']}")
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "script_sha256": audit.digest(Path(__file__)),
        "helper_sha256": audit.digest(helper_path),
        "inputs_sha256": audit.digest(inputs_path),
        "repository": str(repo),
        "output": str(output),
        "reference_directory": str(reference) if reference else None,
        "git_version": subprocess.check_output([git, "--version"], text=True).strip(),
        "gzip_version": subprocess.check_output(
            [compressor, "--version"], text=True, stderr=subprocess.STDOUT
        ).strip(),
        "python_version": sys.version,
        "zlib_compile_version": zlib.ZLIB_VERSION,
        "zlib_runtime_version": zlib.ZLIB_RUNTIME_VERSION,
        "records": [],
    }
    for item in inputs["variants"]:
        name = f"main-source-{item['variant']}.tar.gz"
        target = output / name
        command = [
            git,
            "-C",
            str(repo),
            "archive",
            "--format=tar",
            f"--prefix={item['prefix']}",
            f"--mtime=@{item['mtime']}",
            item["ref"],
        ]
        record = {"variant": item["variant"], "git_command": command, "archive": name}
        with tempfile.TemporaryDirectory(prefix="raw-tar-", dir=output) as temporary:
            raw = Path(temporary) / "source.tar"
            with raw.open("xb") as stream:
                process = subprocess.Popen(
                    command, stdout=stream, stderr=subprocess.PIPE
                )
                error, timed_out = finish(process)
            record["git_returncode"] = process.returncode
            record["git_timed_out"] = timed_out
            record["git_stderr"] = error.decode()
            if process.returncode != 0:
                raise RuntimeError(record)
            record["raw_tar_sha256"] = audit.digest(raw)
            if record["raw_tar_sha256"] != item["expected_tar_sha256"]:
                raise RuntimeError("Generated tar differs before compression")
            with target.open("xb") as stream:
                if item["variant"] == "base":
                    compression_command = [compressor, "-n"]
                    with raw.open("rb") as source:
                        process = subprocess.Popen(
                            compression_command,
                            stdin=source,
                            stdout=stream,
                            stderr=subprocess.PIPE,
                        )
                        error, timed_out = finish(process)
                    record["compression_command"] = compression_command
                    record["gzip_returncode"] = process.returncode
                    record["gzip_timed_out"] = timed_out
                    record["gzip_stderr"] = error.decode()
                    if process.returncode != 0:
                        raise RuntimeError(record)
                else:
                    filename = name.removesuffix(".gz")
                    record["compression_recipe"] = {
                        "implementation": "Python gzip.GzipFile",
                        "filename": filename,
                        "compresslevel": 9,
                        "mtime": 0,
                    }
                    with (
                        raw.open("rb") as source,
                        gzip.GzipFile(
                            filename=filename,
                            mode="wb",
                            compresslevel=9,
                            fileobj=stream,
                            mtime=0,
                        ) as compressed,
                    ):
                        shutil.copyfileobj(source, compressed, length=1024 * 1024)
                    record["python_compression_completed"] = True
        record.update(audit.unpacked_identity(target))
        record["archive_sha256"] = audit.digest(target)
        record["archive_bytes"] = target.stat().st_size
        record["exact_archive_match"] = (
            record["archive_sha256"] == item["expected_archive_sha256"]
            and record["archive_bytes"] == item["expected_archive_bytes"]
        )
        record["uncompressed_tar_match"] = (
            record["tar_sha256"] == item["expected_tar_sha256"]
        )
        record["unpacked_content_and_mode_match"] = (
            record["canonical_members_sha256"]
            == item["expected_canonical_members_sha256"]
        )
        record["timestamp_and_commit_metadata_match"] = (
            record["member_mtimes"] == [item["mtime"]]
            and record["global_pax_headers"].get("comment")
            == item["expected_global_pax_comment"]
        )
        if reference is not None:
            original = reference / name
            record["reference_matches_expected"] = (
                audit.digest(original) == item["expected_archive_sha256"]
            )
            record["byte_equal_reference"] = audit.byte_equal(target, original)
        record["passed"] = all(
            record[name]
            for name in (
                "exact_archive_match",
                "uncompressed_tar_match",
                "unpacked_content_and_mode_match",
                "timestamp_and_commit_metadata_match",
            )
        ) and all(
            record.get(name, True)
            for name in ("reference_matches_expected", "byte_equal_reference")
        )
        report["records"].append(record)
        (output / "exact-results.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(record), flush=True)
    report["all_exact"] = all(record["passed"] for record in report["records"])
    (output / "exact-results.json").write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if report["all_exact"] else 1)


if __name__ == "__main__":
    main()
