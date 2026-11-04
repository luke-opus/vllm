# Native scalar OpenMP worker validation

This fork-only job attempts a normal CPU build of the unchanged OpenMP PR head
`dfdc85947fe914d04906f5d0206c020c208bd400` on the RISE standard
`ubuntu-24.04-riscv` runner. The provider currently routes that label to its
Scaleway EM-RV1 pool. Allocation and actual CPU information must be checked in
the run; source routing alone is not evidence of execution.

The documented CMake option `-DVLLM_RVV_VLEN=0` selects the scalar backend.
Genuine dependencies are installed with uv, including the published RISE Torch
wheel and the original `requirements/cpu.txt`. Rust and build tools are installed
normally. No package version is fabricated, no production backend is replaced,
and any installation/build failure retains its actual exit and log.

After building, `native-openmp-pair.py` checks that the production C++ and CMake
files equal the pinned baseline, then reuses that single real extension for both
policies. It switches only the original one-line `CpuArchEnum.RISCV` addition in
a disposable source copy. Each policy starts the real default CPU MRV2 through
`LLM`, the multiprocess executor, and a fresh spawned worker. Both run the same
six hash-checked Pythia-14m files, two prompts, four tokens each, two CPU cores and
two Torch/OpenMP threads. `taskset` limits each child to those two CPUs;
libgomp can bind its master thread to just one of them. The observer checks live extension mappings, actual
worker state and startup environment, returned output fields, and shutdown.
The source copy and extension identity are checked after every case.

`observer-origin.diff` records the adaptation of the existing model observer:
the model path, thread count, and provenance are parameterized, and the expected
libgomp CPU list uses its actual space-separated format. The original
and adapted hashes are in `observer-origin.json`.

Reproduction uses `.github/workflows/native-openmp-worker.yml` on branch
`riscv-native-openmp-worker`, or runs `bash native-runtime/bootstrap.sh ROOT
PAYLOAD` on a suitable native Ubuntu 24.04 RISC-V host with passwordless apt.
`PAYLOAD` must be an absolute copy of this directory and `ROOT` a fresh absolute
working directory. The dependency versions resolved, build commands, machine
information, child exits and original logs appear in the job output.

This is an attempted native scalar worker check, not a completed result. It
does not validate RVV/ERF, model quality, throughput, thread utilization or
independent physical attestation. The K3/RVV jobs and upstream/human review
requirements remain separate. No assertion is skipped or relaxed to report a
pass.
