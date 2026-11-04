# BF16 model evaluation supplement

This is a new evaluation of the existing, unchanged PR heads. It does not replace
the earlier eight-token smoke result or its disclosed output difference.

The workload is frozen before execution: the first 64 rows of the WikiText-2 raw
test split with at least 64 Pythia tokenizer tokens, truncated to 64 tokens. No
sample selection depends on model outputs. The exact dataset revision, Parquet
size and LFS SHA256 are recorded in `dataset-file.json`. The private
`corpus-inputs.json` holds text and token IDs for execution; public reproduction
should use the preparation script and `corpus-selection.json`, without copying
the original article text into the evidence repository.

Each run invokes the real `LLM.generate` entry point with prompt log probabilities
and the ordinary default CPU MRV2 runner. Two variants differ only by the complete
baseline/fixed production extension. A separate reference uses the baseline
extension and FP32 dtype; it is not called the patched BF16 result. Source trees,
weights, source restoration, live extension maps, worker class, sampling settings,
CPU affinity, process exits, and shutdown are checked and recorded.

The metrics are target-token mean negative log likelihood, perplexity, top-1
next-token accuracy, and target-logprob deviations from the FP32 reference. There
are 4032 scored positions per run. This is a descriptive, finite-subset evaluation
of Pythia-14m on RISC-V Linux/QEMU. It does not prove broad model quality,
full-vocabulary equivalence, performance, native hardware, ERF path coverage, or
OpenMP worker behavior. No threshold is relaxed to obtain a pass, and numerical
changes remain visible even if a reported metric gets worse.

The original current-main runtime helpers and observer are reused unchanged.
Only disposable runtime snapshots may temporarily link the selected extension;
production builds, earlier runs, package versions, and PR sources stay intact.
Any failed attempt must retain its original drivers, input hash, logs and actual
exit before a separately identified correction is tried.
