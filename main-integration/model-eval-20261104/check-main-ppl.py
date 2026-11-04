"""Score frozen test tokens through the real default CPU LLM entry point."""

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--corpus-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--variant", choices=("base", "fixed", "reference-fp32"), required=True
    )
    parser.add_argument("--expected-library", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()
    assert digest(args.corpus) == args.corpus_sha256
    corpus = json.loads(args.corpus.read_text())
    assert len(corpus["samples"]) == 64 and corpus["context_tokens"] == 64
    assert corpus["expected_scored_tokens"] == 4032
    assert not os.environ.get("LD_PRELOAD")
    assert "VLLM_USE_V2_MODEL_RUNNER" not in os.environ
    assert os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] == "0"

    import torch
    import vllm
    import vllm._C
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.platforms import CpuArchEnum, current_platform

    assert current_platform.get_cpu_architecture() == CpuArchEnum.RISCV
    assert Path(vllm.__file__).resolve() == args.source / "vllm/__init__.py"
    assert Path(vllm._C.__file__).resolve() == args.expected_library
    assert digest(args.expected_library) == args.expected_sha256
    torch.set_num_threads(1)
    model = Path("/home/vllmtest/vllm-validation/pythia-14m")
    assert digest(model / "tokenizer.json") == corpus["tokenizer_sha256"]
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    for sample in corpus["samples"]:
        assert (
            tokenizer.encode(sample["text"], add_special_tokens=False)[:64]
            == sample["token_ids"]
        )

    # Reuse the existing observer; it does not replace or patch the worker.
    observer_path = Path(__file__).parent / "check-main-model.py"
    spec = importlib.util.spec_from_file_location(
        "existing_model_observer", observer_path
    )
    observer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(observer)
    dtype = "float32" if args.variant == "reference-fp32" else "bfloat16"
    parameters = {
        "model": str(model),
        "dtype": dtype,
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
    sampling = {"temperature": 0.0, "max_tokens": 1, "prompt_logprobs": 1}
    llm = None
    result = None
    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as profile:
            llm = LLM(**parameters)
            requests = llm.generate(
                [
                    {"prompt_token_ids": sample["token_ids"]}
                    for sample in corpus["samples"]
                ],
                SamplingParams(**sampling),
                use_tqdm=False,
            )
        assert len(requests) == 64
        rows = []
        values = []
        top1 = 0
        for sample, request in zip(corpus["samples"], requests):
            assert list(request.prompt_token_ids) == sample["token_ids"]
            steps = request.prompt_logprobs
            assert steps is not None and len(steps) == 64 and steps[0] is None
            logprobs = []
            ranks = []
            for token, step in zip(sample["token_ids"][1:], steps[1:]):
                assert step is not None and token in step
                value = step[token]
                assert math.isfinite(value.logprob) and value.logprob <= 0
                assert value.rank is not None and value.rank >= 1
                logprobs.append(value.logprob)
                ranks.append(value.rank)
            assert len(logprobs) == 63
            values.extend(logprobs)
            top1 += sum(rank == 1 for rank in ranks)
            rows.append(
                {
                    "row": sample["row"],
                    "token_ids_sha256": sample["token_ids_sha256"],
                    "target_logprobs": logprobs,
                    "target_ranks": ranks,
                    "negative_log_likelihood_sum": -math.fsum(logprobs),
                    "generated_token_ids": list(request.outputs[0].token_ids),
                }
            )
        assert len(values) == 4032
        (worker,) = llm.collective_rpc(observer.worker_state, timeout=120)
        assert worker["pid"] == os.getpid() and worker["device"] == "cpu"
        assert worker["torch_threads"] == 1 and worker["use_v2_model_runner"] is True
        assert (
            worker["model_runner_class"]
            == "vllm.v1.worker.cpu.standalone_model_runner.StandaloneCPUModelRunner"
        )
        assert worker["extension_sha256"] == args.expected_sha256
        assert worker["LD_PRELOAD"] == worker["startup_LD_PRELOAD"] == ""
        operators = {
            event.key: event.count
            for event in profile.key_averages()
            if event.key.startswith("_C::")
        }
        assert operators.get("_C::cpu_attention_with_kv_cache", 0) > 0
        assert operators.get("_C::cpu_attn_reshape_and_cache", 0) > 0
        nll = -math.fsum(values) / len(values)
        result = {
            "variant": args.variant,
            "parameters": parameters,
            "sampling_parameters": sampling,
            "corpus_sha256": args.corpus_sha256,
            "dataset_revision": corpus["dataset"]["revision"],
            "model": "EleutherAI/pythia-14m",
            "model_revision": "cf967c0a9a04383db6f7b1108d86b2962634b4ac",
            "samples": rows,
            "metrics": {
                "samples": len(rows),
                "scored_tokens": len(values),
                "mean_negative_log_likelihood": nll,
                "perplexity": math.exp(nll),
                "top1_correct": top1,
                "top1_accuracy": top1 / len(values),
            },
            "worker": worker,
            "operator_counts": operators,
            "observer_sha256": digest(observer_path),
            "vllm_distribution_metadata": observer.distribution_metadata(),
            "scope": "Fixed 64x64-token WikiText-2 test subset on QEMU; target-token scores, not full-vocabulary equivalence or broad model quality/performance proof.",
        }
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()
    assert result is not None
    result["shutdown_returned"] = True
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "variant": args.variant,
                "metrics": result["metrics"],
                "shutdown_returned": True,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
