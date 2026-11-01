"""Capture real trained-model outputs and CPU operator events for one C++ build."""

import argparse
import json
import math
import os
from pathlib import Path


def worker_state(self):
    import torch
    from vllm.v1.worker.cpu_worker import CPUWorker

    assert isinstance(self.worker, CPUWorker)
    maps = Path("/proc/self/maps").read_text().splitlines()
    return {
        "class": type(self.worker).__module__ + "." + type(self.worker).__name__,
        "pid": os.getpid(),
        "device": str(self.device),
        "torch_threads": torch.get_num_threads(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "LD_PRELOAD": os.environ.get("LD_PRELOAD", ""),
        "loaded_openmp": sorted(
            {
                line.split()[-1]
                for line in maps
                if "libgomp" in line or "libomp.so" in line
            }
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=("base", "fixed"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = root / "bf16-model-comparison"

    import torch
    from vllm import LLM, SamplingParams
    from vllm.platforms import CpuArchEnum, current_platform

    assert current_platform.get_cpu_architecture() == CpuArchEnum.RISCV
    assert not os.environ.get("LD_PRELOAD")
    torch.set_num_threads(1)
    parameters = {
        "model": str(root / "pythia-14m"),
        "dtype": "bfloat16",
        "max_model_len": 128,
        "max_num_seqs": 2,
        "max_num_batched_tokens": 128,
        "block_size": 32,
        "enforce_eager": True,
        "tensor_parallel_size": 1,
        "distributed_executor_backend": "uni",
        "kv_cache_memory_bytes": 64 * 1024 * 1024,
        "seed": 0,
        "enable_prefix_caching": False,
        "enable_chunked_prefill": False,
    }
    llm = None
    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as profile:
            llm = LLM(**parameters)
            requests = llm.generate(
                ["The capital of France is", "A computer can"],
                SamplingParams(temperature=0.0, max_tokens=4, logprobs=5),
                use_tqdm=False,
            )
        assert len(requests) == 2
        outputs = []
        for request in requests:
            assert len(request.outputs) == 1
            item = request.outputs[0]
            assert len(item.token_ids) == 4
            assert item.logprobs is not None and len(item.logprobs) == 4
            assert all(
                math.isfinite(value.logprob)
                for step in item.logprobs
                for value in step.values()
            )
            outputs.append(
                {
                    "prompt": request.prompt,
                    "prompt_token_ids": list(request.prompt_token_ids),
                    "token_ids": list(item.token_ids),
                    "text": item.text,
                    "cumulative_logprob": item.cumulative_logprob,
                    "finish_reason": item.finish_reason,
                    "stop_reason": item.stop_reason,
                    "logprobs": [
                        {
                            str(token): {
                                "logprob": value.logprob,
                                "rank": value.rank,
                                "decoded_token": value.decoded_token,
                            }
                            for token, value in step.items()
                        }
                        for step in item.logprobs
                    ],
                }
            )
        (worker,) = llm.collective_rpc(worker_state, timeout=120)
        assert worker["pid"] == os.getpid()
        assert worker["device"] == "cpu" and worker["torch_threads"] == 1
        assert worker["LD_PRELOAD"] == ""
        operators = {
            event.key: event.count
            for event in profile.key_averages()
            if event.key.startswith("_C::")
        }
        assert operators.get("_C::cpu_attention_with_kv_cache", 0) > 0
        assert operators.get("_C::cpu_attn_reshape_and_cache", 0) > 0
        result = {
            "variant": args.variant,
            "model": "EleutherAI/pythia-14m",
            "revision": "cf967c0a9a04383db6f7b1108d86b2962634b4ac",
            "parameters": parameters,
            "worker": worker,
            "operator_counts": operators,
            "outputs": outputs,
            "scope": "QEMU smoke comparison; all returned top-5/chosen-token logprobs, not full vocabulary; inline BF16 conversion not directly instrumented",
        }
        data = json.dumps(result, indent=2, allow_nan=False) + "\n"
        (output_dir / f"{args.variant}.json").write_text(data)
        print(data, flush=True)
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
