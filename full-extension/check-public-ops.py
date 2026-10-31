"""Exercise actual operators registered by the complete CPU extension."""

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("library")
parser.add_argument("--output", type=Path, default=Path("public-ops-results.json"))
args = parser.parse_args()
torch.set_num_threads(1)
torch.ops.load_library(args.library)
results = {}
for dtype in (torch.float32, torch.bfloat16):
    first = torch.tensor(
        [
            -3.0,
            -2.0,
            -1.0,
            -0.0,
            0.0,
            0.125,
            0.5,
            1.0,
            2.0,
            3.0,
            4.0,
            -4.0,
            0.25,
            -0.25,
            0.75,
            -0.75,
        ],
        dtype=dtype,
    ).reshape(1, -1)
    second = torch.ones_like(first)
    inputs = torch.cat((first, second), dim=-1).contiguous()
    for approximate in ("none", "tanh"):
        output = torch.empty_like(first)
        operator = (
            torch.ops._C.gelu_and_mul
            if approximate == "none"
            else torch.ops._C.gelu_tanh_and_mul
        )
        operator(output, inputs)
        reference = functional.gelu(first, approximate=approximate) * second
        finite = bool(torch.isfinite(output).all())
        atol = 1e-5 if dtype == torch.float32 else 1e-3
        rtol = 1.3e-6 if dtype == torch.float32 else 1.6e-2
        try:
            torch.testing.assert_close(output, reference, atol=atol, rtol=rtol)
            matches = True
        except AssertionError:
            matches = False
        results[f"gelu_{dtype!s}_{approximate}"] = {
            "finite": finite,
            "matches_reference": matches,
            "nan_count": int(torch.isnan(output).sum().item()),
        }
values = torch.full((1, 128), 0.125, dtype=torch.bfloat16)
inputs = torch.cat((values, torch.ones_like(values)), dim=-1).contiguous()
output = torch.empty_like(values)
torch.ops._C.silu_and_mul(output, inputs)
expected = functional.silu(values.float()).to(torch.bfloat16)
results["silu_bf16"] = {
    "exact_match": bool(torch.equal(output, expected)),
    "output_bits": hex(output.view(torch.uint16)[0, 0].item()),
    "expected_bits": hex(expected.view(torch.uint16)[0, 0].item()),
}
args.output.write_text(json.dumps(results, indent=2) + "\n")
try:
    moe_input = torch.full((4, 128), 0.5, dtype=torch.bfloat16)
    identity = torch.eye(128, dtype=torch.bfloat16)
    w13 = torch.cat((identity, identity), dim=0).unsqueeze(0).contiguous()
    w2 = identity.unsqueeze(0).contiguous()
    packed_w13 = torch.empty_like(w13)
    packed_w2 = torch.empty_like(w2)
    torch.ops._C.prepack_moe_weight(w13, packed_w13, "vec")
    torch.ops._C.prepack_moe_weight(w2, packed_w2, "vec")
    moe_output = torch.empty_like(moe_input)
    torch.ops._C.cpu_fused_moe(
        moe_output,
        moe_input,
        packed_w13,
        packed_w2,
        None,
        None,
        torch.ones((4, 1), dtype=torch.float32),
        torch.zeros((4, 1), dtype=torch.int32),
        False,
        "gelu",
        "vec",
    )
    moe_reference = (
        functional.gelu(moe_input.float(), approximate="none") * moe_input.float()
    ).to(torch.bfloat16)
    results["moe_gelu_bf16"] = {
        "finite": bool(torch.isfinite(moe_output).all()),
        "nan_count": int(torch.isnan(moe_output).sum().item()),
        "exact_match": bool(torch.equal(moe_output, moe_reference)),
    }
except RuntimeError as error:
    results["moe_gelu_bf16"] = {"error": str(error)}
args.output.write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps(results, indent=2), flush=True)
