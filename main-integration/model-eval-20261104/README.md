# BF16 fixed-subset language-model evaluation, 2026-11-04

This is an additional evaluation of the unchanged BF16 draft head
`6592612016059859dff1e93bd534edc1a5737615` against the original baseline
`e0d00e4b8eb0c89ec30921d0695cfdd409e9c211`. It uses the same complete production
extensions and documented shared build adaptations as the existing
[`main-integration`](../README.md) evidence. It runs under RISC-V Linux/QEMU,
not on native hardware. No package version, model activation, weight or
production backend is replaced for this evaluation.

The workload was frozen before execution: the first 64 WikiText-2 raw test rows
with at least 64 Pythia tokenizer tokens, truncated to 64 tokens each. This
gives 4,032 scored next-token positions per variant. The original dataset
revision, bytes, license information and LFS SHA256 are in `dataset-file.json`.
`corpus-selection.json` records the selected row numbers, text/token hashes and
exact private input hash. The original article text and token inputs are not
redistributed here. Selection does not depend on any model output.

Three serial fresh processes use the real `LLM.generate` prompt-logprob entry
point, ordinary default `StandaloneCPUModelRunner`, pinned public Pythia-14m
weights, seed 0, TP=1, uni execution, eager mode and one CPU thread. Base/fixed
BF16 differ only in the complete extension. The separate reference uses the
baseline extension and FP32 dtype; it is not labeled the patched BF16 result.
The real worker, production operator counts, loaded library maps and hashes,
model/tokenizer inputs, actual child exits and shutdown are verified. Each
temporary source/extension change is restored and independently audited.

See `summary.json` for negative log likelihood, perplexity, target-token top-1
accuracy and target-logprob MAE/RMSE/maximum absolute difference from FP32.
`prepare-public-evidence.py` independently recalculates all metrics from the
recorded 4,032 target scores and checks actual exits, identities and restoration
before preparing this evidence set. There is no success threshold for an
improvement in model quality; numerical changes are retained even when a metric
worsens. Original logs are lossless gzip archives, with uncompressed hashes in
`summary.json` and archive hashes in `manifest.json`.

For reproduction, first follow the existing full-library build/runtime setup in
`main-integration/build` and `main-integration/runtime`. Keep the original
`main_runtime_common.py`, `check-main-model.py`, `run-main-model-pair.py` and
`supervise-main.py`; their executed hashes are in `pair-results.json`. Copy
the three new runtime drivers (`check-main-ppl.py`, `run-main-ppl-pair.py`,
`supervise-main-ppl.py`) into the guest's existing
`/home/vllmtest/vllm-validation/main-integration-runtime` directory.

Prepare inputs on a host using uv and a dedicated environment with
`pyarrow==26.0.1` and `tokenizers==0.23.3`. Put the preparation files in
`work/bf16-evaluation-20261104`, the pinned Parquet file at
`work/bf16-evaluation-20261104/wikitext-2-test.parquet`, and the pinned Pythia
tokenizer at `work/riscv-payload/pythia-14m/tokenizer.json`. For example:

```sh
uv venv --python 3.12 work/bf16-evaluation-20261104/tools-venv
uv pip install --python work/bf16-evaluation-20261104/tools-venv/bin/python \
  pyarrow==26.0.1 tokenizers==0.23.3
work/bf16-evaluation-20261104/tools-venv/bin/python \
  work/bf16-evaluation-20261104/prepare-corpus.py
```

Check the
generated selection/input hash against `corpus-selection.json`, then copy the
private `corpus-inputs.json` into the guest runtime directory. All input download
URLs and hashes are in the existing model manifest and dataset record.

Run from the guest runtime directory with the genuine existing environment:

```sh
../main-venv/bin/python supervise-main-ppl.py --execute \
  --run-id main-bf16-wikitext-01 \
  --corpus /home/vllmtest/vllm-validation/main-integration-runtime/corpus-inputs.json \
  --corpus-sha256 3d4702a0be87574ffcd3788869cceb000b1d3f720e170a0beb74031d37f90a2c
```

Use a fresh run ID if that output directory already exists; never overwrite an
earlier attempt. The frozen plan, actual driver logs, per-variant output arrays,
pair report and independent external-exit/restoration audit are retained here.

This is a descriptive evaluation of one small model on one finite test subset.
It does not establish broad model quality, full-vocabulary equivalence, ERF path
coverage, native RVV execution, throughput or upstream/human approval. Earlier
eight-token smoke differences remain disclosed separately.
