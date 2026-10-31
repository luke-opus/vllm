"""Apply the same existing PyTorch BLAS fallback to three independent builds."""

import hashlib
import io
import json
import shutil
import tarfile
from pathlib import Path

root = Path(__file__).resolve().parent
payload = root.parent / "riscv-payload"
relative = "CMakeFiles/_C.dir/csrc/cpu/sgl-kernels/fla.cpp.o"
replacement = (root / "vllm-validation/build-base" / relative).read_bytes()
command = json.loads((root / "compile-one-command.json").read_text())
assert command["cross_command"][-1] == "-UVLLM_HAS_OPENBLAS"
assert command["cross_command"][-2].endswith("/sgl-kernels/fla.cpp")
assert replacement[:6] == b"\x7fELF\x02\x01"
assert int.from_bytes(replacement[18:20], "little") == 243
base_object = (root / "baseline-source-objects" / relative).read_bytes()
refs = {
    "base": "ca8dbccf204abfcdf9dcb178f7e108c06b224b0f",
    "erf": "7f1353f1a5854545163d49f764552a57f836b2b2",
    "bf16": "174e56854703c0b9ede89e0a41f9e04dc4c59bc1",
}
for variant, ref in refs.items():
    objects = {}
    if variant == "base":
        directory = root / "baseline-source-objects"
        objects = {
            "build-base/" + str(path.relative_to(directory)): path.read_bytes()
            for path in directory.rglob("*.o")
        }
    else:
        with tarfile.open(payload / f"cross-{variant}-objects.tar.gz") as source:
            objects = {m.name: source.extractfile(m).read() for m in source}
    assert len(objects) == 19
    assert objects["build-base/" + relative] == base_object
    objects["build-base/" + relative] = replacement
    archive = payload / f"cross-{variant}-reference-blas-objects.tar.gz"
    with tarfile.open(archive, "w:gz", compresslevel=1) as bundle:
        for name, data in objects.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            bundle.addfile(info, io.BytesIO(data))
    manifest = {
        "variant": variant,
        "source_ref": ref,
        "objects": len(objects),
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "build_mode": "existing PyTorch reference BLAS fallback",
        "override": command,
        "replaced_object": relative,
        "original_object_sha256": hashlib.sha256(base_object).hexdigest(),
        "replacement_object_sha256": hashlib.sha256(replacement).hexdigest(),
    }
    (payload / f"cross-{variant}-reference-blas-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(variant, manifest["sha256"], archive.stat().st_size)
shutil.copyfile(
    root / "link-reference-blas-on-guest.py",
    payload / "link-reference-blas-on-guest.py",
)
