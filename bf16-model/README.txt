Real Pythia-14m BF16 base/fixed model comparison (2026-11-01)

Both real LLM/CPUWorker runs completed on RISC-V Linux/QEMU: base exit 0 in
211.60 seconds, fixed exit 0 in 216.12 seconds. The driver also exited 0.
There was no timeout. Raw outputs and logs are base.json/base.txt and
fixed.json/fixed.txt; these are newly paired runs, not reused HF results.

Model: EleutherAI/pythia-14m, revision
cf967c0a9a04383db6f7b1108d86b2962634b4ac, unchanged public trained weights.
Base C++: ca8dbccf204abfcdf9dcb178f7e108c06b224b0f.
Fixed C++: 174e56854703c0b9ede89e0a41f9e04dc4c59bc1 (BF16-only PR).
Python model-source/cpu.py, weights, configuration, environment, executor and
OpenMP linkage stay constant. Only the complete extension symlink changes.

                         base tokens             fixed tokens
The capital of France is 253,1986,2077,15          253,1986,2077,13
A computer can           320,908,281,897           320,908,281,897

Seven of eight generated tokens match. The first text ends in a period on
base and a comma on fixed. Both outputs are finite. There are 39 common
returned token logprobs across the eight steps with identical input prefixes:
maximum absolute difference 0.3876380920410156, mean 0.0811952367807046.
One of eight top-5 sets changes (second generated step of the first prompt).
These are descriptive differences, not an accuracy-improvement or numerical-
equivalence claim. All returned top-5/chosen-token logprobs are preserved;
this is not a full-vocabulary distribution capture.

Both profiles observe 24 cpu_attention_with_kv_cache calls and 24
cpu_attn_reshape_and_cache calls. Both logs select the Transformers modeling
backend and retain the existing oneDNN-linear-to-Torch-linear fallback.
coverage.txt gives precise fixed-ref source links relating the RVV attention
output's BF16Vec16::save to float_to_bf16. The inline helper and runtime ISA
argument were not directly instrumented. This does not establish native-
silicon correctness, broad accuracy, ERF model coverage, upstream CI or human
test/review confirmation.

Provenance and restoration:
- object-provenance.json and inventory.txt link both object archives to the
  prior full-extension manifests; object-differences.json records each object.
- comparison.json records the complete base link command, shared oneDNN hash,
  library hashes, constant environment and actual child return codes.
- dynamic-base.txt/dynamic-fixed.txt show identical NEEDED and RUNPATH entries.
- independent-audit.json is a separate post-run inspection: archive hashes
  match the prior manifests; the reused fixed DSO matches the prior public
  openmp-worker/openmp-worker-run-results.json constant_library_sha256
  (a13aadd18a4f7d9ccf275f983e12d425b92965d33fc0b9e39b65029f9af570ad).
  These external checks are not assertions retroactively added to the runner.
- maps-observations.jsonl captures the fixed extension in its actual live
  process. Base extension live maps were not captured before process exit.
  Both worker JSON records contain the same GNU/LLVM OpenMP mappings.
- The independent audit read the actual restored symlink and cpu.py bytes;
  cpu.py matches git base SHA bfa74a5c08f4186b57a3bb83cf1c08fbe1bf35b5522c02020fc1d602dca12f50.
- executed-script-sha256.txt matches the unchanged local check-model.py and
  run-comparison.py copies that were executed. driver-exit.json records the
  observed outer command/driver completion.

Reproduction in the previously provisioned guest:
1. Use the existing full-extension and runtime-dependencies setup, pinned
   model-files.json files, base model-source and two reference-BLAS object
   archives. The existing fixed LLVM-OpenMP DSO must also be present.
2. Put check-model.py, run-comparison.py, native-link-commands.json and
   model-files.json under ~/vllm-validation/bf16-model-comparison/.
3. From ~/vllm-validation:
   .venv/bin/python -u bf16-model-comparison/run-comparison.py
4. With both result JSON files beside analyze-results.py:
   .venv/bin/python bf16-model-comparison/analyze-results.py

The captured runner is retained as executed. Its unused timeout branch sets
a driver status of 124 after killing the child; that value would not establish
an actual child exit of 124. The two actual child exits here were both zero.
audit-restoration.py is a separate post-run verifier for this captured run;
it expects the separately collected live maps record as well as the results.
