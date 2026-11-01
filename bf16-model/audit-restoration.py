"""Independently inspect original source restoration and recorded build hashes."""

import hashlib
import json
import os
import time
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    root = Path.home() / "vllm-validation"
    out = root / "bf16-model-comparison"
    comparison = json.loads((out / "comparison.json").read_text())
    assert [item["returncode"] for item in comparison["variants"]] == [0, 0]
    expected = {
        "cross-base-reference-blas-objects.tar.gz": "371fdfe466354dc07ba54b1385dc43cf593db481e4240625af81c89a7c0d9b2a",
        "cross-bf16-reference-blas-objects.tar.gz": "b0a44f813e7ba39f28af7bebd159d8ca3a5edad44c3778ee370e5dde11c95510",
        "build-base/_C-bf16-linked-openmp.abi3.so": "a13aadd18a4f7d9ccf275f983e12d425b92965d33fc0b9e39b65029f9af570ad",
        "model-source/vllm/platforms/cpu.py": "bfa74a5c08f4186b57a3bb83cf1c08fbe1bf35b5522c02020fc1d602dca12f50",
    }
    hashes = {name: digest(root / name) for name in expected}
    assert hashes == expected, hashes
    expected_link = str(root / "build-base/_C-bf16-linked-openmp.abi3.so")
    actual_link = os.readlink(root / "model-source/vllm/_C.abi3.so")
    assert actual_link == expected_link
    assert (
        comparison["constant_cpu_sha256"]
        == hashes["model-source/vllm/platforms/cpu.py"]
    )
    for item in comparison["variants"]:
        assert digest(Path(item["library"])) == item["library_sha256"]
    observations = [
        json.loads(line)
        for line in (out / "maps-observations.jsonl").read_text().splitlines()
    ]
    fixed_maps = [
        item
        for snapshot in observations
        for item in snapshot["observations"]
        if "fixed" in item["argv"]
        and any(expected_link in line for line in item["maps"])
    ]
    assert fixed_maps, "Expected a live fixed-process extension mapping observation"
    report = {
        "checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "expected_hash_sources": {
            "archives": "prior complete-build cross-base/cross-bf16-reference-blas-manifest.json",
            "fixed_dso": "prior complete LLVM OpenMP link record supplied by parent reviewer",
            "cpu_py": "git show ca8dbccf204abfcdf9dcb178f7e108c06b224b0f:vllm/platforms/cpu.py",
        },
        "expected_sha256": expected,
        "observed_sha256": hashes,
        "all_expected_hashes_match": True,
        "current_symlink": actual_link,
        "original_symlink_restored": True,
        "original_cpu_source_restored": True,
        "both_library_hashes_still_match_run_record": True,
        "fixed_extension_observed_in_live_process_maps": True,
        "base_extension_live_maps": "not captured before base process exited",
        "checks_scope": "Independent post-run inspection, separate from assertions in run-comparison.py",
    }
    (out / "independent-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
