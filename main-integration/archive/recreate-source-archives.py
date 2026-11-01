"""Recreate the four frozen source archives without overwriting existing files."""

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def unpacked_identity(path):
    canonical = hashlib.sha256()
    members = 0
    files = 0
    mtimes = set()
    with tarfile.open(path, mode="r|gz") as archive:
        for member in archive:
            mtimes.add(member.mtime)
            record = {
                "name": member.name,
                "type": member.type.decode("ascii"),
                "mode": member.mode,
                "linkname": member.linkname,
                "size": member.size,
            }
            if member.isfile():
                stream = archive.extractfile(member)
                record["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
                files += 1
            canonical.update(
                json.dumps(
                    record, sort_keys=True, separators=(",", ":"), ensure_ascii=True
                ).encode()
                + b"\n"
            )
            members += 1
        global_pax_headers = dict(archive.pax_headers)
    with gzip.open(path, "rb") as stream:
        tar_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "tar_sha256": tar_sha256,
        "canonical_members_sha256": canonical.hexdigest(),
        "members": members,
        "regular_files": files,
        "member_mtimes": sorted(mtimes),
        "global_pax_headers": global_pax_headers,
    }


def byte_equal(left, right):
    with left.open("rb") as a, right.open("rb") as b:
        while True:
            first = a.read(1024 * 1024)
            second = b.read(1024 * 1024)
            if first != second:
                return False
            if not first:
                return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument(
        "--inputs",
        type=Path,
        default=Path(__file__).with_name("source-archive-recreation-inputs.json"),
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output.resolve()
    reference = args.reference_dir.resolve() if args.reference_dir else None
    if reference is not None and output == reference:
        parser.error("Output must not be the reference directory")
    if output.exists() and any(output.iterdir()):
        parser.error("Output directory must be new or empty")
    git = shutil.which("git")
    compressor = shutil.which("gzip")
    if git is None or compressor is None:
        parser.error("Real git and gzip executables are required")
    inputs = json.loads(args.inputs.read_text())
    for item in inputs["variants"]:
        kind = subprocess.check_output(
            [git, "-C", str(repo), "cat-file", "-t", item["ref"]], text=True
        ).strip()
        if kind != item["object_type"]:
            parser.error(f"Wrong/missing frozen object type for {item['variant']}")
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "script_sha256": digest(Path(__file__)),
        "inputs_sha256": digest(args.inputs),
        "repository": str(repo),
        "output": str(output),
        "reference_directory": str(reference) if reference else None,
        "git_executable": git,
        "gzip_executable": compressor,
        "git_version": subprocess.check_output([git, "--version"], text=True).strip(),
        "gzip_version": subprocess.check_output(
            [compressor, "--version"], text=True, stderr=subprocess.STDOUT
        ).strip(),
        "records": [],
    }
    report_path = output / "recreation-results.json"
    for item in inputs["variants"]:
        name = f"main-source-{item['variant']}.tar.gz"
        path = output / name
        git_command = [
            git,
            "-C",
            str(repo),
            "archive",
            "--format=tar",
            f"--prefix={item['prefix']}",
            f"--mtime=@{item['mtime']}",
            item["ref"],
        ]
        gzip_command = [compressor, "-n"]
        record = {
            "variant": item["variant"],
            "archive": name,
            "git_command": git_command,
            "gzip_command": gzip_command,
            "timed_out": False,
        }
        git_error = output / f"{item['variant']}-git-stderr.txt"
        gzip_error = output / f"{item['variant']}-gzip-stderr.txt"
        with (
            path.open("xb") as target,
            git_error.open("w") as ge,
            gzip_error.open("w") as ze,
        ):
            producer = subprocess.Popen(git_command, stdout=subprocess.PIPE, stderr=ge)
            consumer = subprocess.Popen(
                gzip_command, stdin=producer.stdout, stdout=target, stderr=ze
            )
            producer.stdout.close()
            try:
                record["gzip_returncode"] = consumer.wait(timeout=300)
                record["git_returncode"] = producer.wait(timeout=30)
            except subprocess.TimeoutExpired:
                record["timed_out"] = True
                for process in (consumer, producer):
                    if process.poll() is None:
                        process.kill()
                record["gzip_returncode"] = consumer.wait()
                record["git_returncode"] = producer.wait()
        success = (
            record["git_returncode"] == record["gzip_returncode"] == 0
            and not record["timed_out"]
        )
        if success:
            record.update(unpacked_identity(path))
            record["archive_sha256"] = digest(path)
            record["archive_bytes"] = path.stat().st_size
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
                and record["members"] == item["expected_members"]
                and record["regular_files"] == item["expected_regular_files"]
            )
            record["timestamp_and_commit_metadata_match"] = (
                record["member_mtimes"] == [item["mtime"]]
                and record["global_pax_headers"].get("comment")
                == item["expected_global_pax_comment"]
            )
            if reference is not None:
                original = reference / name
                record["reference_sha256"] = digest(original)
                record["reference_matches_expected"] = (
                    record["reference_sha256"] == item["expected_archive_sha256"]
                )
                record["byte_equal_reference"] = byte_equal(path, original)
            record["passed"] = bool(
                record["exact_archive_match"]
                and record["uncompressed_tar_match"]
                and record["unpacked_content_and_mode_match"]
                and record["timestamp_and_commit_metadata_match"]
                and record.get("reference_matches_expected", True)
                and record.get("byte_equal_reference", True)
            )
        else:
            record["passed"] = False
        report["records"].append(record)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(record), flush=True)
    report["all_exact"] = all(r["passed"] for r in report["records"])
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if report["all_exact"] else 1)


if __name__ == "__main__":
    main()
