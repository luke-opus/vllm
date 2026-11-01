"""Check actual RISC-V configuration and child-process library inheritance."""

import json
import os
import subprocess
import sys
from pathlib import Path

import torch
from vllm.config import DeviceConfig, VllmConfig
from vllm.platforms import CpuArchEnum, current_platform
from vllm.platforms.cpu import CpuPlatform

assert sys.platform.startswith("linux")
assert current_platform.get_cpu_architecture() == CpuArchEnum.RISCV
assert not os.environ.get("LD_PRELOAD"), "Start without a forced preload"
root = Path.home() / "vllm-validation"
config = VllmConfig(device_config=DeviceConfig(device="cpu"))
CpuPlatform.check_and_update_config(config)
preload = os.environ.get("LD_PRELOAD", "")
library = Path(torch.__file__).parent / "lib/libgomp.so.1"
assert library.is_file()
expected_fixed = sys.argv[1] == "fixed"
assert (str(library) in preload.split(":")) == expected_fixed, preload
child = r"""
import json
import os
from pathlib import Path
import torch
torch.set_num_threads(1)
root = Path.home() / "vllm-validation"
torch.ops.load_library(str(root / "build-base/_C-bf16-linked-openmp.abi3.so"))
x = torch.full((1, 128), 0.125, dtype=torch.bfloat16)
out = torch.empty_like(x)
torch.ops._C.silu_and_mul(out, torch.cat((x, torch.ones_like(x)), dim=-1))
assert int(out.view(torch.uint16)[0, 0]) == 0x3d88
maps = Path("/proc/self/maps").read_text().splitlines()
print(json.dumps({"LD_PRELOAD": os.environ.get("LD_PRELOAD", ""),
"threads": torch.get_num_threads(), "silu_bits": hex(int(out.view(torch.uint16)[0, 0])),
"loaded_openmp": sorted({line.split()[-1] for line in maps if "libgomp" in line or "libomp.so" in line})}))
"""
completed = subprocess.run(
    [sys.executable, "-c", child],
    env=os.environ.copy(),
    check=True,
    capture_output=True,
    text=True,
    timeout=900,
)
data = json.loads(completed.stdout.splitlines()[-1])
assert data["LD_PRELOAD"] == preload
assert data["threads"] == 1
assert any("libgomp" in path for path in data["loaded_openmp"])
result = {
    "variant": sys.argv[1],
    "architecture": "actual riscv64 Linux/QEMU",
    "preload": preload,
    "child": data,
    "scope": "real configuration and child operator/library inheritance; not a vLLM worker, native silicon, or performance proof",
}
(root / f"native-openmp-config-{sys.argv[1]}.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
print(json.dumps(result, indent=2), flush=True)
