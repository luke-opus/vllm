"""Fetch and verify the pinned public model without using account credentials."""

import hashlib
import json
import tarfile
from pathlib import Path
from urllib.request import urlopen

root = Path.home() / "vllm-validation"
base = "http://10.0.2.2:8765/"
manifest = json.loads(urlopen(base + "pythia-14m-manifest.json", timeout=30).read())
assert manifest["revision"] == "cf967c0a9a04383db6f7b1108d86b2962634b4ac"
archive = root / "pythia-14m.tar.gz"
with urlopen(base + archive.name, timeout=120) as response:
    archive.write_bytes(response.read())
assert hashlib.sha256(archive.read_bytes()).hexdigest() == manifest["sha256"]
with tarfile.open(archive) as bundle:
    members = bundle.getmembers()
    assert len(members) == manifest["files"] == 6
    assert all(member.isfile() and member.name.startswith("pythia-14m/") for member in members)
    bundle.extractall(root, filter="data")
script = urlopen(base + "check-real-model.py", timeout=30).read()
(root / "check-real-model.py").write_bytes(script)
print("Verified pinned Pythia-14m model and saved real LLM validation script", flush=True)
