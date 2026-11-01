"""Package exactly the successful fresh-database objects with provenance."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    args = parser.parse_args()
    data = json.loads(args.record.read_text())
    records = data["records"]
    config = data["config"]
    assert len(records) == data["expected_units"]
    assert all(r["returncode"] == 0 and not r["timed_out"] for r in records)
    assert len({r["object"] for r in records}) == len(records)
    build = Path(config["mirror"]) / "build" / config["variant"]
    archive = args.record.parent / f"objects-{config['variant']}.tar.gz"
    with tarfile.open(archive, "w:gz", compresslevel=1) as bundle:
        for record in records:
            path = build / record["object"]
            assert digest(path) == record["object_sha256"]
            bundle.add(path, arcname=record["object"])
    manifest = {
        "package_script_sha256": digest(Path(__file__)),
        "variant": config["variant"],
        "source_identity": config["source_identity"],
        "archive_sha256": digest(archive),
        "archive_bytes": archive.stat().st_size,
        "objects": {r["object"]: r["object_sha256"] for r in records},
        "compile_record_sha256": digest(args.record),
        "database_sha256": digest(build / "compile_commands.json"),
    }
    (args.record.parent / f"objects-{config['variant']}.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
