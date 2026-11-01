"""Run the real LLM entry point and compare a pinned trained model with HF."""

import json
import math
from pathlib import Path

import torch
from tests.models.utils import check_logprobs_close
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.logprobs import Logprob

root = Path.home() / "vllm-validation"
model_path = root / "pythia-14m"
torch.set_num_threads(1)
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
prompts = ["The capital of France is", "A computer can"]
reference_model = AutoModelForCausalLM.from_pretrained(
    model_path, dtype=torch.bfloat16, local_files_only=True
).eval()
reference_outputs = []
with torch.inference_mode():
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        generated = reference_model.generate(
            **inputs,
            max_new_tokens=4,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        ids = generated.sequences[0, inputs["input_ids"].shape[-1] :].tolist()
        logprobs = []
        for score in generated.scores:
            values, indices = score[0].float().log_softmax(-1).topk(5)
            assert bool(torch.isfinite(values).all())
            logprobs.append(
                {
                    index: Logprob(logprob=value, rank=rank + 1)
                    for rank, (index, value) in enumerate(
                        zip(indices.tolist(), values.tolist())
                    )
                }
            )
        reference_outputs.append((ids, tokenizer.decode(ids), logprobs))
del reference_model
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU]
) as profile:
    llm = LLM(
        model=str(model_path),
        dtype="bfloat16",
        max_model_len=128,
        max_num_seqs=2,
        max_num_batched_tokens=128,
        block_size=32,
        enforce_eager=True,
        tensor_parallel_size=1,
        distributed_executor_backend="uni",
        kv_cache_memory_bytes=64 * 1024 * 1024,
        seed=0,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
    )
    requests = llm.generate(
        prompts,
        SamplingParams(temperature=0.0, max_tokens=4, logprobs=5),
        use_tqdm=False,
    )
actual_outputs = []
for request in requests:
    output = request.outputs[0]
    assert len(output.token_ids) == 4
    assert all(
        math.isfinite(item.logprob)
        for step in output.logprobs
        for item in step.values()
    )
    actual_outputs.append((list(output.token_ids), output.text, output.logprobs))
check_logprobs_close(
    outputs_0_lst=reference_outputs,
    outputs_1_lst=actual_outputs,
    name_0="HF pinned Pythia-14m BF16",
    name_1="vLLM RISC-V BF16",
)
operators = {
    event.key: event.count
    for event in profile.key_averages()
    if event.key.startswith("_C::")
}
result = {
    "model": "EleutherAI/pythia-14m",
    "revision": "cf967c0a9a04383db6f7b1108d86b2962634b4ac",
    "scope": "Two-prompt greedy smoke comparison, four tokens each; not native hardware or an accuracy benchmark",
    "reference_tokens": [item[0] for item in reference_outputs],
    "vllm_tokens": [item[0] for item in actual_outputs],
    "operator_counts": operators,
    "reference_comparison": "check_logprobs_close passed",
}
(root / "real-model-results.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)
