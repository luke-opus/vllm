"""Observe OpenMP startup state through a real spawned vLLM CPUWorker."""

import argparse
import json
import math
import os
from pathlib import Path


def worker_state(self):
    import os
    from pathlib import Path

    import torch
    from vllm.v1.worker.cpu_worker import CPUWorker

    assert isinstance(self.worker, CPUWorker)
    startup = dict(
        entry.split(b"=", 1)
        for entry in Path("/proc/self/environ").read_bytes().split(b"\0")
        if b"=" in entry
    )
    maps = Path("/proc/self/maps").read_text().splitlines()
    return {
        "pid": os.getpid(),
        "worker_class": type(self.worker).__module__ + "." + type(self.worker).__name__,
        "device": str(self.device),
        "LD_PRELOAD": os.environ.get("LD_PRELOAD", ""),
        "startup_LD_PRELOAD": startup.get(b"LD_PRELOAD", b"").decode(),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", ""),
        "GOMP_CPU_AFFINITY": os.environ.get("GOMP_CPU_AFFINITY", ""),
        "OMP_PLACES": os.environ.get("OMP_PLACES", ""),
        "OMP_PROC_BIND": os.environ.get("OMP_PROC_BIND", ""),
        "torch_num_threads": torch.get_num_threads(),
        "affinity": sorted(os.sched_getaffinity(0)),
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
    parser.add_argument("--root", type=Path, default=Path.home() / "vllm-validation")
    args = parser.parse_args()
    root = args.root.resolve()
    assert not os.environ.get("LD_PRELOAD"), (
        "Launch the parent without a forced preload"
    )
    assert os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] == "0"

    import torch
    from vllm import LLM, SamplingParams
    from vllm.platforms import CpuArchEnum, current_platform

    assert current_platform.get_cpu_architecture() == CpuArchEnum.RISCV
    library = Path(torch.__file__).parent / "lib/libgomp.so.1"
    assert library.is_file()
    parent_pid = os.getpid()
    llm = LLM(
        model=str(root / "pythia-14m"),
        dtype="bfloat16",
        max_model_len=128,
        max_num_seqs=2,
        max_num_batched_tokens=128,
        block_size=32,
        enforce_eager=True,
        tensor_parallel_size=1,
        distributed_executor_backend="mp",
        kv_cache_memory_bytes=64 * 1024 * 1024,
        seed=0,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
    )
    try:
        requests = llm.generate(
            ["The capital of France is", "A computer can"],
            SamplingParams(temperature=0.0, max_tokens=4, logprobs=5),
            use_tqdm=False,
        )
        assert len(requests) == 2
        outputs = []
        for request in requests:
            assert len(request.outputs) == 1
            output = request.outputs[0]
            assert len(output.token_ids) == 4
            assert output.logprobs is not None and len(output.logprobs) == 4
            assert all(
                math.isfinite(item.logprob)
                for step in output.logprobs
                for item in step.values()
            )
            outputs.append(
                {
                    "prompt": request.prompt,
                    "prompt_token_ids": list(request.prompt_token_ids),
                    "token_ids": list(output.token_ids),
                    "text": output.text,
                    "cumulative_logprob": output.cumulative_logprob,
                    "finish_reason": output.finish_reason,
                    "stop_reason": output.stop_reason,
                    "logprobs": [
                        {
                            str(token): {
                                "logprob": item.logprob,
                                "rank": item.rank,
                                "decoded_token": item.decoded_token,
                            }
                            for token, item in step.items()
                        }
                        for step in output.logprobs
                    ],
                }
            )
        (worker,) = llm.collective_rpc(worker_state, timeout=120)
        result = {
            "variant": args.variant,
            "model": "EleutherAI/pythia-14m",
            "revision": "cf967c0a9a04383db6f7b1108d86b2962634b4ac",
            "architecture": "actual riscv64 Linux/QEMU",
            "parent_pid": parent_pid,
            "expected_libgomp": str(library),
            "worker": worker,
            "outputs": outputs,
            "scope": "One spawned CPUWorker, two prompts and four tokens each; no physical hardware or performance claim",
        }
        (root / f"openmp-worker-{args.variant}.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n"
        )
        print(json.dumps(result, indent=2, allow_nan=False), flush=True)
        assert worker["pid"] != parent_pid
        assert worker["device"] == "cpu"
        assert worker["torch_num_threads"] == 1
        assert worker["OMP_NUM_THREADS"] == "1"
        assert any(
            Path(path).resolve() == library.resolve()
            for path in worker["loaded_openmp"]
        )
        if args.variant == "fixed":
            assert worker["LD_PRELOAD"] == str(library)
            assert worker["startup_LD_PRELOAD"] == str(library)
            assert (
                worker["GOMP_CPU_AFFINITY"] == os.environ["VLLM_CPU_OMP_THREADS_BIND"]
            )
        else:
            assert worker["LD_PRELOAD"] == ""
            assert worker["startup_LD_PRELOAD"] == ""
    finally:
        llm.llm_engine.engine_core.shutdown()
    print(f"{args.variant}: real CPUWorker and model checks passed", flush=True)


if __name__ == "__main__":
    main()
