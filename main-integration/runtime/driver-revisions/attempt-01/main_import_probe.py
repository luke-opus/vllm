"""Read-only pytest plugin recording imports in the actual repository test process."""

import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path


def observation():
    import torch
    import vllm
    import vllm._C

    source = Path(os.environ["MAIN_VALIDATION_SOURCE"])
    expected_library = Path(os.environ["MAIN_VALIDATION_LIBRARY"])
    extension = Path(vllm._C.__file__).resolve()
    assert Path(sys.prefix) == Path("/home/vllmtest/vllm-validation/main-venv")
    assert Path(vllm.__file__).resolve() == source / "vllm/__init__.py"
    assert extension == expected_library
    with extension.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    assert sha == os.environ["MAIN_VALIDATION_LIBRARY_SHA256"]
    maps = [
        line
        for line in Path("/proc/self/maps").read_text().splitlines()
        if "_C.abi3.so" in line
    ]
    assert any(Path(line.split()[-1]).resolve() == extension for line in maps)
    try:
        dist = importlib.metadata.distribution("vllm")
    except importlib.metadata.PackageNotFoundError:
        metadata = {"present": False}
    else:
        metadata = {
            "present": True,
            "version": dist.version,
            "location": str(dist.locate_file("")),
            "direct_url": dist.read_text("direct_url.json"),
        }
    return {
        "pid": os.getpid(),
        "sys_prefix": sys.prefix,
        "vllm_file": vllm.__file__,
        "vllm_version_string": vllm.__version__,
        "vllm_distribution_metadata": metadata,
        "torch_file": torch.__file__,
        "extension_file": vllm._C.__file__,
        "extension_resolved": str(extension),
        "extension_sha256": sha,
        "extension_maps": maps,
    }


def pytest_sessionstart(session):
    Path(os.environ["MAIN_VALIDATION_IMPORT_RECORD"]).write_text(
        json.dumps({"start": observation()}, indent=2) + "\n"
    )


def pytest_sessionfinish(session, exitstatus):
    path = Path(os.environ["MAIN_VALIDATION_IMPORT_RECORD"])
    data = json.loads(path.read_text())
    data.update(finish=observation(), pytest_exit_status=int(exitstatus))
    path.write_text(json.dumps(data, indent=2) + "\n")
