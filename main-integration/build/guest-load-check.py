"""Load the complete extension and exercise the newly required CPU operators."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant")
    args = parser.parse_args()
    root = Path.home() / "vllm-validation/main-integration-build"
    results = root / "results" / args.variant
    library = root / "build" / args.variant / "_C.abi3.so"

    import torch

    torch.set_num_threads(1)
    torch.ops.load_library(str(library))
    table = torch.tensor([[3, 4], [5, 6]], dtype=torch.int32)
    gathered = torch.full_like(table, -1)
    mapping = torch.tensor([1], dtype=torch.int64)
    blocks = torch.tensor([2, 2], dtype=torch.int32)
    torch.ops._C.gather_block_table_cpu(table, gathered, mapping, blocks, 2)
    assert gathered.tolist() == [[5, 6], [0, 0]]
    input_ids = torch.empty(2, dtype=torch.int32)
    lookahead = torch.empty((1, 1), dtype=torch.int32)
    torch.ops._C.prepare_prefill_inputs_cpu(
        input_ids,
        lookahead,
        torch.tensor([0], dtype=torch.int64),
        torch.tensor([0, 2], dtype=torch.int32),
        torch.tensor([[10, 11, 12]], dtype=torch.int32),
        torch.tensor([3], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
    )
    assert input_ids.tolist() == [10, 11] and lookahead.tolist() == [[12]]
    logits = torch.tensor([[1.0, 3.0, 2.0], [9.0, 0.0, 1.0]])
    sampled = torch.ops._C.gumbel_sample_cpu(
        logits,
        torch.tensor([0, 1], dtype=torch.int64),
        torch.zeros(2),
        torch.tensor([10, 20], dtype=torch.int64),
        torch.tensor([0, 0], dtype=torch.int64),
        True,
        False,
        False,
    )
    assert sampled.tolist() == [1, 0]
    with library.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    maps = [
        line
        for line in Path("/proc/self/maps").read_text().splitlines()
        if str(library) in line or "libomp.so" in line or "libgomp" in line
    ]
    assert any(str(library) in line for line in maps)
    report = {
        "variant": args.variant,
        "torch_version": torch.__version__,
        "torch_file": torch.__file__,
        "library_sha256": sha,
        "gathered": gathered.tolist(),
        "input_ids": input_ids.tolist(),
        "next_prefill_tokens": lookahead.tolist(),
        "greedy_tokens": sampled.tolist(),
        "loaded_maps": maps,
        "scope": "Real complete DSO load and three new CPU operator calls; not repository/model validation",
    }
    (results / "load-check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
