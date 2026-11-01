Pinned-main runtime preparation and prepared drivers
==================================================

Pinned upstream main: e0d00e4b8eb0c89ec30921d0695cfdd409e9c211.
This directory is separate from all earlier-base validation evidence.

Environment preparation completed
---------------------------------
Guest: /home/vllmtest/vllm-validation/main-venv.
Copied real packages with cp -a --reflink=auto from the old .venv, not a hard-link
copy. 34,124 common regular files were checked: zero shared regular inodes.
The copied interpreter reports main-venv as sys.prefix and /usr as base_prefix.
Use main-venv/bin/python and python -m pytest; copied console-script shebangs and
activation scripts are not trusted to relocate automatically.

Installed only the real PyPI transformers 5.22.0 universal wheel using uv,
offline/no-index, UV_LINK_MODE=copy, with the wheel and cache inside the new
runtime directory. Official PyPI metadata, wheel, METADATA, WHEEL and provenance
are preserved. The downloaded wheel SHA-256 is:
3cca65f94b0aa33511433c7c1a661c513e4ad8bc0504cb370d181d771e7c9aa0.

The only changed distribution is transformers 5.20.2 -> 5.22.0.
uv pip check: 153 packages, all compatible. Actual imports and calls passed:
Torch 2.14.0+cpu BF16 tensor operation; Transformers AutoConfig/AutoTokenizer/
AutoModelForCausalLM imports; XGrammar and LLGuidance regex APIs; Harmony native
module; pytest and pytest_asyncio imports. No model was loaded by these probes.
The first probe used the distribution name as an import name for Harmony and
failed; its script/log is retained. The corrected real openai_harmony module and
native extension imported successfully without another package change.

Evidence: old-before.json/old-after.json and freezes, copied-before.json,
copy-audit.json, new-after.json/freeze, final-package-audit.json, pip-check.txt,
install-transformers.txt, runtime-check.json/txt, inode-isolation.json.
The old environment inventory and freeze were unchanged. All 9,439 checked Torch
headers/critical files were unchanged; there are 9,433 headers. This is a selected
CPU runtime/test dependency environment, not installation of the complete common
or CUDA test requirement lists. No full-suite readiness is claimed.

importlib.metadata.distribution('vllm') actually raises PackageNotFoundError in
main-venv; vllm-distribution-metadata.json records that result without importing
vllm or loading any DSO. Drivers use an explicit PYTHONPATH into a disposable copy
of the pinned source and will record real vllm.__file__, actual loaded _C path,
resolved DSO path/hash and live /proc maps for both model variants. No version
metadata or _version.py is manufactured. Source archive development-version
warnings are retained in child logs.

Drivers prepared, NOT EXECUTED
-----------------------------
main_runtime_common.py provides exact input/three-library gates, independent
source staging, controlled environment, transactional file restoration, and real
subprocess exit/timeout recording. All three link/load-check records must exist
and match the frozen source identities and actual DSO hashes before any suite.
Sources are read only in main-integration-build/source; each driver creates a new
copy inside main-integration-runtime/runs. Old source/build/env are untouched.

run-main-repository.py runs real pytest from the copied repository root, retaining
tests/conftest.py and all normal fixtures. It uses exactly the same candidate test
files on base/fixed while retaining the matching production headers and full DSO.
main_import_probe.py is a read-only pytest plugin: it records and verifies actual
vllm/Torch/module/DSO imports and live DSO maps in that pytest process, at session
start and finish. It does not replace fixtures, mark tests or change tolerances.
Expected baseline failures remain failures in pytest, explicitly reported as
regression demonstrations, never counted as passing tests:
  ERF: 4 selected cases, expected base 2 failures / fixed 0.
  BF16 operator: 1 case, expected base 1 failure / fixed 0.
  BF16 source-header boundary: 18 cases, expected base 9 failures / fixed 0.
  OpenMP config: 6 cases, expected base 2 failures / fixed 0.
Unexpected failure type, skipped test, collection/setup error, timeout, or exit
code fails the driver. Base production headers are never overwritten by candidate
headers. The boundary fixture is the genuine new repository fixture and invokes
the guest compiler, so wait for the build agent's compilation to finish.

The standalone suite runs both full files on base/ERF/BF16:
  tests/v1/worker/cpu/test_cpu_block_table.py
  tests/v1/worker/cpu/test_cpu_input_batch.py
plus exactly:
  tests/v1/worker/cpu/test_cpu_gumbel_sample.py::test_greedy_supports_dtypes_strides_and_invalid_mapping
  tests/v1/worker/cpu/test_cpu_gumbel_sample.py::test_pre_scaled_logits_match_temperature_application
No marks or tolerances are changed. Actual JUnit test/skip/error/failure counts are
reported; collection alone is not a pass. The large statistical gumbel cases are
not selected.

run-main-model-pair.py / check-main-model.py keep the default MRV2 selection:
VLLM_USE_V2_MODEL_RUNNER is absent, and real public LLM.collective_rpc observes the
actual CPUWorker and StandaloneCPUModelRunner. Both use the same frozen real
Pythia-14m model files (six SHA-256/size checks) and generation settings.
BF16 comparison keeps base Python source and environment constant and switches
only the independent complete base/BF16 DSO. Both live process maps and DSO hashes
are captured. Attention/cache profiler events are required. Statistics describe
returned top-5/chosen tokens; aggregate shared-token logprob deltas include only
steps with identical generated prefixes. No full-vocabulary, accuracy improvement,
or numerical equivalence is asserted.
OpenMP keeps the complete new-main base DSO constant and only swaps the two exact
frozen cpu.py files. It checks the one-line RISCV membership difference, real
spawned mp worker PID, startup LD_PRELOAD from /proc/self/environ, bundled OpenMP
mapping, effective affinity and exact equality of all returned output fields.
The previous main's cpu.py is never used. The current six-case test file is frozen
separately. These are QEMU smoke checks, not physical hardware or performance.

supervise-main.py adds an external driver exit record and independent byte/link/
mode restoration checks. Child and driver timeouts retain timed_out and the actual
OS return code, never replace it with 124. SIGTERM/SIGINT ask for cleanup; a forced
SIGKILL, host failure or power loss cannot guarantee Python finally execution and
must fail the external restoration audit if state remains changed. Successful
model result JSON is written only after shutdown returned; report completion also
requires restoration and successful child exit. No timeout/recovery experiment has
yet been run on these new drivers.

Only syntax/static checks have been run on the prepared drivers. Root review,
all three genuine DSO load gates, build-compiler scheduling, and explicit root
authorization precede execution. Proposed execution from the guest runtime dir:
  ../main-venv/bin/python supervise-main.py --execute --kind repository --run-id main-repository-01
  ../main-venv/bin/python supervise-main.py --execute --kind bf16 --run-id main-bf16-model-01
  ../main-venv/bin/python supervise-main.py --execute --kind openmp --run-id main-openmp-worker-01
These examples are not executed by preparing this directory.
