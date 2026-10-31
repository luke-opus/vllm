"""Package real source headers and the proposed regression for an offline target."""

import argparse
import io
import subprocess
import tarfile
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("checkout", type=Path)
parser.add_argument("base_ref")
parser.add_argument("fixed_ref")
parser.add_argument("--output", type=Path, default=Path("results/bf16-pytest.tar.gz"))
args = parser.parse_args()
test = (Path(__file__).resolve().parent / "test_cpu_float_conversion.py").read_bytes()
args.output.parent.mkdir(parents=True, exist_ok=True)
with tarfile.open(args.output, "w:gz") as archive:
    for name, ref in (("base", args.base_ref), ("fixed", args.fixed_ref)):
        header = subprocess.check_output(
            [
                "git",
                "-C",
                str(args.checkout),
                "show",
                f"{ref}:csrc/cpu/float_convert.hpp",
            ]
        )
        for relative, data in (
            ("csrc/cpu/float_convert.hpp", header),
            ("tests/kernels/core/test_cpu_float_conversion.py", test),
        ):
            info = tarfile.TarInfo(f"{name}/{relative}")
            info.size = len(data)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(data))
print(args.output)
