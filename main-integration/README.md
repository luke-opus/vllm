# Pinned-main RISC-V draft validation

This evidence set concerns upstream `e0d00e4b8eb0c89ec30921d0695cfdd409e9c211`
and three independent fixes. It does not combine ERF and BF16 changes in one
library. The target is Ubuntu 24.04 RISC-V Linux under QEMU 11.1.2, RVV VLEN 256
with agnostic lanes filled with ones. It is not physical-hardware or upstream-CI
evidence. Main can advance after this pinned snapshot.

| PR | Exact tested source commit | Frozen source tree |
| --- | --- | --- |
| [#64078](https://github.com/vllm-project/vllm/pull/64078) | `ec0fbf5e9b3e9d796ccae667a437f2e97c3f9cde` | `94453352d881e58ebf423b67d7ccea2ba449eef5` |
| [#64079](https://github.com/vllm-project/vllm/pull/64079) | `6592612016059859dff1e93bd534edc1a5737615` | `8b860fc3eba99f6eccfe4b473c00a829b7ea9af3` |
| [#64080](https://github.com/vllm-project/vllm/pull/64080) | `dfdc85947fe914d04906f5d0206c020c208bd400` | `2cc5383ab7ca9209554779029edb9ccc31a41432` |


The source heads preserve each existing PR's history with a normal main merge
and a follow-up commit. Their trees exactly match the frozen build inputs.
The 18-case BF16 header test, its CI selection, documentation clarifications,
and removal of the obsolete OpenMP test exemption are included in those trees.

## Build and documentation

All three fresh configurations have 231 compilation units: 21 vLLM production
units plus 210 oneDNN units. All **63 production units were independently
recompiled**, and three complete links and three genuine load checks exited 0.
Each load check calls the real block-table, prefill-input and greedy-sampling
operators added on the pinned main. The old oneDNN static library was reused
only after all 210 compile commands, generated configuration, Torch headers
and its recorded SHA256 matched; it was not freshly recompiled here.

See [build instructions and outcomes](build/README.txt),
[machine-readable build summary](build/build-summary.json), and original logs.
Shared environment adaptations remain explicit: the oneDNN `std::isfinite`
declaration include, real LLVM OpenMP linkage, and the existing Torch BLAS
fallback selected for `fla.cpp` because the installed OpenBLAS lacks `sbgemm_`.
No dummy symbol, replacement backend or lazy loader is used.

Original-config changed-file hooks and normal commit/sign-off hooks passed. Three complete
strict MkDocs builds with API generation passed, each recording 3,026 API
entries. See [check results](source/candidate-check-results.json), raw build
logs and [independent local documentation discovery](source/docs-local-fresh.txt).
Independent [GitHub-source](source/docs-github-fresh.txt) and
[public-preview](source/docs-web-fresh.txt) checks record their actual navigation
and any search limits. The [final checklist](source/pr-checklist-final.txt)
separates completed work from outstanding requirements.
The official NumPy inventory was cached after verification; strict checks and
the repository configuration were retained. Existing INFO-level link diagnostics
remain in the logs. Independent previews do not establish upstream RTD success.

The [current-head CI audit](source/current-head-ci-audit.txt) records DCO success
and the upstream contributor-eligibility pre-run failure. Format/lint checks were
skipped. Each RTD build stopped in `docs/pre_run_check.sh` on that same failed
check before dependency installation or MkDocs execution. These are genuine
unsatisfied upstream gates, not successful CI or an observed MkDocs error.

## Complete manual pre-commit on Linux

The earlier local changed-file checks are now supplemented by the full original
CI command on four fresh GitHub-hosted Ubuntu 24.04 x86-64 jobs:

```sh
uvx pre-commit run --show-diff-on-failure --color=always --all-files --hook-stage manual
```

[Run 41283200026](https://github.com/luke-opus/vllm/actions/runs/41283200026)
completed successfully on 2026-11-01. The [raw fork records](fork-precommit/)
preserve original artifacts, logs, source and configuration identities, and
actual exits. Each job's 33 selected manual hooks passed, with no failed or
skipped selected hooks. All four pre-commit exits are 0 and all four final Git
status/diff files are empty. The baseline and candidate source identities are
the exact commits/trees above. No hook was disabled and no configuration or
generated lockfile/formatting changes were pushed back into the source PRs.

The run's workflow commit is
`90a7bbc95260d2c889880e636e79d145ce5e8e4e`; the tested commits are independently
verified checkouts recorded in each artifact, not inferred from that workflow
head. The artifact ZIP SHA256 values match GitHub's digests, extracted bytes
match the ZIP entries, and configuration hashes match the original Git objects
before and after execution. Docker was available for the graph-generation hook.
The original manual stage naturally excludes the configuration's local Rust
and commit-message hooks; existing source DCO/local-hook evidence remains separate.

This is supplemental execution in the user's fork on hosted Linux, not the
upstream self-hosted runner. The upstream contributor-eligibility gate and RTD
status described above remain unresolved; this run also supplies no native
RISC-V, model or human-review evidence.

## Actual repository tests

| Selected group | Tests | Passed | Failed | Errors / skips |
| --- | ---: | ---: | ---: | ---: |
| erf-base | 4 | 2 | 2 | 0 / 0 |
| erf-erf | 4 | 4 | 0 | 0 / 0 |
| bf16-operator-base | 1 | 0 | 1 | 0 / 0 |
| bf16-boundary-base | 18 | 9 | 9 | 0 / 0 |
| bf16-operator-bf16 | 1 | 1 | 0 | 0 / 0 |
| bf16-boundary-bf16 | 18 | 18 | 0 | 0 / 0 |
| openmp-config-base | 6 | 4 | 2 | 0 / 0 |
| openmp-config-openmp | 6 | 6 | 0 | 0 / 0 |
| standalone-base | 24 | 24 | 0 | 0 / 0 |
| standalone-erf | 24 | 24 | 0 | 0 / 0 |
| standalone-bf16 | 24 | 24 | 0 | 0 / 0 |

Total: **130 executions, 116 passes and 14 expected baseline assertion failures; no errors or skips.** All 11 comparison groups completed as specified; the driver and independent supervisor exited 0 and all restoration checks passed.

These runs retain the genuine current root `tests/conftest.py` and fixtures.
The same candidate tests run against independent base/fixed production headers
and complete libraries. Expected baseline assertion failures remain pytest
failures and are explicitly distinguished from passing fixed tests. Every
selected run reports actual collection, failures, errors, skips and OS exit;
neither collection alone nor a prepared driver is a passing result.

The read-only pytest probe checks the source path, resolved extension, SHA256
and live process maps at session start and finish. Temporary test files, source
files and library links are restored, then independently audited by the
supervisor. See the [repository result](runtime/runs/main-repository-01/repository-results.json)
and [external exit/restoration audit](runtime/supervision/main-repository-01/driver-exit-and-restoration.json).

## Model and worker observations

Both comparisons use real pinned `EleutherAI/pythia-14m` weights at
`cf967c0a9a04383db6f7b1108d86b2962634b4ac`, two prompts, four greedy tokens
each, BF16, seed 0, TP=1, eager execution, block size 32 and a 64 MiB KV cache.
They use the actual default standalone CPU MRV2 path without a runner override.
The separate environment contains real Transformers 5.22.0; the original
environment remains unchanged. Selected runtime dependencies are validated,
not the complete CUDA/common test requirement list.

### BF16 independent-library comparison

Both real `LLM` children completed and shut down, each with actual exit 0;
the outer driver and independent restoration audit also passed. The actual
`CPUWorker` uses `StandaloneCPUModelRunner` (`use_v2=true`). Base Python source,
model weights and parameters remain fixed; only the complete independent DSO
changes. Both workers have affinity `[0, 1]`, one Torch thread and no startup
preload. Both live DSO mappings and SHA256 values were verified.

**7/8 greedy output tokens match.** The first completion changes its final token
from period to comma (`15` to `13`); the second is identical. For 39 common
returned logprobs at identical prefixes, the maximum absolute difference is
**0.3876304626464844** and the mean is **0.08119453527988532**. One of eight
returned top-5 sets changes. These are descriptive observations, not equivalence
or model-quality evidence. Each side records 24 attention and 24 cache profiler
events, plus the real standalone input/block/sampling operator calls. Handler
creation is not claimed as proof of a particular oneDNN forward kernel.
Both logs report that oneDNN cannot create the linear matmul primitive and the
production code falls back to Torch linear. These model observations include
that real fallback; no successful oneDNN linear execution is claimed.

See [pair result](runtime/runs/main-bf16-model-01/pair-results.json),
[comparison detail](runtime/runs/main-bf16-model-01/comparison-details.json),
[base](runtime/runs/main-bf16-model-01/base.txt),
[fixed](runtime/runs/main-bf16-model-01/fixed.txt), and
[external audit](runtime/supervision/main-bf16-model-01/driver-exit-and-restoration.json).


### OpenMP real spawned-worker comparison

Both `LLM` children completed generation and shutdown with actual exit 0;
the outer driver and independent restoration audit passed. The two runs use
real `mp`/`spawn` CPU workers and the default `StandaloneCPUModelRunner`, holding
the newly built main baseline DSO constant. Only the exact one-line RISC-V
membership change in the current platform policy differs.

All returned output fields, including all eight tokens and their logprobs, are
exactly equal. The fixed worker starts with PyTorch's real bundled `libgomp` in
`/proc/self/environ`; the baseline starts without a preload. Actual worker PIDs,
source/DSO identities, process maps, thread counts and affinities are recorded.
Both workers use one Torch thread with affinity `[0]`, and both maps still
contain LLVM OpenMP and GNU OpenMP. Both model logs retain the production
oneDNN-to-Torch linear fallback.
This establishes startup inheritance and output preservation in this setup,
not a reproduced baseline model failure, performance gain, or elimination of
multiple OpenMP runtimes.

The first attempt, `main-openmp-model-01`, failed before worker launch because
the Unix IPC socket path exceeded the system limit. Actual exit 1 and successful
restoration are preserved; that attempt is not a passing OpenMP test. Revision
02 only shortens the isolated temporary directory. Model settings, production
source, DSO inputs and assertions remain unchanged. Repository/BF16 successful
runs still map to their original driver revision.

See [successful pair](runtime/runs/main-openmp-model-02/pair-results.json),
[base](runtime/runs/main-openmp-model-02/base.txt),
[fixed](runtime/runs/main-openmp-model-02/fixed.txt),
[external audit](runtime/supervision/main-openmp-model-02/driver-exit-and-restoration.json),
[first failed attempt](runtime/runs/main-openmp-model-01/pair-results.json), and
[exact driver revisions](runtime/revision-map.json).


The model drivers record actual source/extension paths, both live maps and
library hashes, worker classes, process identity, preload and outputs. A success
JSON is written only after shutdown returns; the parent records actual child
exit and separately audits restoration. Timeouts retain both their flag and
the real OS return code. Parallel load, when present, is recorded separately;
no elapsed time is a performance benchmark.

Source archives have no installed vLLM distribution metadata; the genuine
development-version warnings are retained. No version metadata is fabricated.
Returned top-5/chosen-token logprobs do not represent full-vocabulary output,
and these small smoke comparisons do not establish model-quality improvement.

## Reproduce and inspect

1. Obtain a Git checkout containing the exact commit/tree objects in the table
   above, or apply the three independent frozen patches in `source/` to the
   pinned base in separate disposable checkouts. Verify each resulting tree.
2. Follow [source archive reconstruction](archive/source-archive-recreation-README.txt).
   All four original archives were rebuilt and compared byte-for-byte. The
   baseline uses `gzip -n`; the candidates use their recorded Python GzipFile
   recipe. The initial compression mismatch and its successful correction are
   both preserved. Tool-version differences must not be silently accepted as
   an exact archive match.
3. Use the real dependency/toolchain provenance in [the earlier complete build](../full-extension/README.md),
   the new [build instructions](build/README.txt), and the isolated
   [runtime instructions](runtime/README.txt). Replace local path mappings as
   needed and use explicit virtual-environment Python interpreters.
4. Run `supervise-main.py --execute --kind repository`, then the BF16 and
   OpenMP comparisons with unique run IDs as documented in the runtime README.
   Keep production inputs independent and retain all raw exits and failures.

The explicit file manifests contain SHA256 and byte counts; raw logs and
scripts are copied unchanged. Libraries, objects, model weights, caches and
credentials are excluded. Reproduction requires their real published inputs;
the manifests do not substitute for a complete library or model run.

## Remaining boundaries

- [Current-main ERF model-route audit](source/erf-model-coverage-current-main.txt)
  did not establish an unchanged public checkpoint exercising gated exact GELU
  through the changed CPU/RVV implementation. Dense/Torch GELU and Nomic's
  `GELU_NO_MUL` route cannot close that gap. The bounded search is not proof that
  no such checkpoint exists anywhere.
- Native RVV hardware and upstream CI have not been established by these local
  records. The separate Cloud-V inventory workflow must actually obtain and
  execute on a runner before any physical-hardware claim.
  The [inactive native header plan](native-plan/README.md) references these exact
  source commits and preserves the earlier plan separately; configuration checks
  are preparation only.
- The submitting human's personal line review and test confirmation are not
  inferred from authorization to continue. The upstream PRs remain drafts;
  this evidence does not request re-review or claim every requirement is met.

Earlier-base evidence remains separately archived outside this directory.
It is not counted as execution of the current pinned-main source.
