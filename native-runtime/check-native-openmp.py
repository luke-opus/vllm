"""Observe the genuine pinned-main LLM and CPUWorker without replacing internals."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path


def worker_state(self):
    import os
    from pathlib import Path

    import torch
    import vllm
    import vllm._C
    from vllm.v1.worker.cpu_worker import CPUWorker

    assert isinstance(self.worker, CPUWorker)
    extension = Path(vllm._C.__file__).resolve()
    with extension.open("rb") as stream:
        extension_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    startup = dict(
        entry.split(b"=", 1)
        for entry in Path("/proc/self/environ").read_bytes().split(b"\0")
        if b"=" in entry
    )
    maps = Path("/proc/self/maps").read_text().splitlines()
    extension_maps = [line for line in maps if "_C.abi3.so" in line]
    assert extension_maps, "The actual extension must appear in live process maps"
    assert any(Path(line.split()[-1]).resolve() == extension for line in extension_maps)
    return {
        "pid": os.getpid(),
        "device": str(self.device),
        "worker_class": type(self.worker).__module__ + "." + type(self.worker).__name__,
        "model_runner_class": type(self.worker.model_runner).__module__
        + "."
        + type(self.worker.model_runner).__name__,
        "use_v2_model_runner": self.worker.use_v2_model_runner,
        "vllm_file": vllm.__file__,
        "torch_file": torch.__file__,
        "extension_file": vllm._C.__file__,
        "extension_resolved": str(extension),
        "extension_sha256": extension_sha,
        "extension_maps": extension_maps,
        "LD_PRELOAD": os.environ.get("LD_PRELOAD", ""),
        "startup_LD_PRELOAD": startup.get(b"LD_PRELOAD", b"").decode(),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", ""),
        "GOMP_CPU_AFFINITY": os.environ.get("GOMP_CPU_AFFINITY", ""),
        "OMP_PLACES": os.environ.get("OMP_PLACES", ""),
        "OMP_PROC_BIND": os.environ.get("OMP_PROC_BIND", ""),
        "torch_threads": torch.get_num_threads(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "loaded_openmp": sorted(
            {
                line.split()[-1]
                for line in maps
                if "libgomp" in line or "libomp.so" in line
            }
        ),
    }


def distribution_metadata():
    try:
        dist = importlib.metadata.distribution("vllm")
    except importlib.metadata.PackageNotFoundError:
        return {"present": False}
    return {
        "present": True,
        "version": dist.version,
        "location": str(dist.locate_file("")),
        "metadata_paths": [
            str(dist.locate_file(p))
            for p in dist.files or []
            if p.name in ("METADATA", "direct_url.json")
        ],
        "direct_url": dist.read_text("direct_url.json"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("bf16", "openmp"), required=True)
    parser.add_argument("--variant", choices=("base", "fixed"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-library", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--threads", type=int, required=True)
    args = parser.parse_args()
    assert args.threads >= 2
    assert not os.environ.get("LD_PRELOAD"), "Parent must start without forced preload"
    assert "VLLM_USE_V2_MODEL_RUNNER" not in os.environ, "Exercise the real default"
    assert os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] == "0"

    import torch
    import vllm
    import vllm._C
    from vllm import LLM, SamplingParams
    from vllm.platforms import CpuArchEnum, current_platform

    assert current_platform.get_cpu_architecture() == CpuArchEnum.RISCV
    assert Path(vllm.__file__).resolve() == args.source / "vllm/__init__.py"
    assert Path(vllm._C.__file__).resolve() == args.expected_library
    torch.set_num_threads(args.threads)
    parameters = {
        "model": str(args.model),
        "dtype": "bfloat16",
        "max_model_len": 128,
        "max_num_seqs": 2,
        "max_num_batched_tokens": 128,
        "block_size": 32,
        "enforce_eager": True,
        "tensor_parallel_size": 1,
        "distributed_executor_backend": "uni" if args.kind == "bf16" else "mp",
        "kv_cache_memory_bytes": 64 * 1024 * 1024,
        "seed": 0,
        "enable_prefix_caching": False,
        "enable_chunked_prefill": False,
    }
    sampling = {"temperature": 0.0, "max_tokens": 4, "logprobs": 5}
    llm = None
    result = None
    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as profile:
            llm = LLM(**parameters)
            requests = llm.generate(
                ["The capital of France is", "A computer can"],
                SamplingParams(**sampling),
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
        assert worker["device"] == "cpu" and worker["torch_threads"] == args.threads
        assert worker["use_v2_model_runner"] is True
        assert worker["model_runner_class"] == (
            "vllm.v1.worker.cpu.standalone_model_runner.StandaloneCPUModelRunner"
        )
        assert Path(worker["vllm_file"]).resolve() == args.source / "vllm/__init__.py"
        assert Path(worker["extension_resolved"]) == args.expected_library
        assert worker["extension_sha256"] == args.expected_sha256
        operators = {
            event.key: event.count
            for event in profile.key_averages()
            if event.key.startswith("_C::")
        }
        bundled_openmp = Path(torch.__file__).parent / "lib/libgomp.so.1"
        if args.kind == "bf16":
            assert worker["pid"] == os.getpid()
            assert worker["LD_PRELOAD"] == worker["startup_LD_PRELOAD"] == ""
            assert operators.get("_C::cpu_attention_with_kv_cache", 0) > 0
            assert operators.get("_C::cpu_attn_reshape_and_cache", 0) > 0
        else:
            assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
            assert worker["pid"] != os.getpid()
            assert worker["OMP_NUM_THREADS"] == str(args.threads)
            assert bundled_openmp.is_file()
            assert any(
                Path(path).resolve() == bundled_openmp.resolve()
                for path in worker["loaded_openmp"]
            )
            preload = str(bundled_openmp) if args.variant == "fixed" else ""
            assert worker["LD_PRELOAD"] == worker["startup_LD_PRELOAD"] == preload
            if args.variant == "fixed":
                expected_cpus = os.environ["VLLM_CPU_OMP_THREADS_BIND"].split(",")
                assert worker["GOMP_CPU_AFFINITY"].split() == expected_cpus
        result = {
            "kind": args.kind,
            "variant": args.variant,
            "policy_base_ref": "e0d00e4b8eb0c89ec30921d0695cfdd409e9c211",
            "build_ref": "dfdc85947fe914d04906f5d0206c020c208bd400",
            "model": "EleutherAI/pythia-14m",
            "revision": "cf967c0a9a04383db6f7b1108d86b2962634b4ac",
            "parameters": parameters,
            "sampling_parameters": sampling,
            "vllm_distribution_metadata": distribution_metadata(),
            "vllm_version_string": vllm.__version__,
            "parent_pid": os.getpid(),
            "worker": worker,
            "parent_profile_operator_counts": operators,
            "outputs": outputs,
            "scope": (
                "RISE native-provider job, default MRV2, two prompts/four tokens each; "
                "returned top-5/chosen-token logprobs only, not full vocabulary; "
                "no quality, performance, or independent physical attestation claim"
            ),
        }
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()
    assert result is not None
    result["shutdown_returned"] = True
    data = json.dumps(result, indent=2, allow_nan=False) + "\n"
    args.output.write_text(data)
    print(data, flush=True)


if __name__ == "__main__":
    main()
