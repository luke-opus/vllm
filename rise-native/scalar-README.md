# Native scalar BF16 validation

The separate `riscv-rise-scalar-validation` branch uses the documented standard
RISE label `ubuntu-24.04-riscv`. The provider maps ordinary repositories to a
Scaleway EM-RV1. No assumption of RVV 1.0 support is made. This is additional
coverage for the production helper shared by the RISC-V scalar and vector
backends; the independent K3 job and its RVV/ERF requirements remain outstanding.

The workflow exports the actual baseline and fixed `float_convert.hpp` from the
same pinned source refs used by the existing draft #64079. It reuses the unchanged
public comparison driver and checks all 1458753 inputs against its independent
nearest-representable-value reference. Expected base exit is 1 with 667067
mismatches; fixed exit must be 0 with zero mismatches. Compilation uses the native
GCC 13 default ISA and does not force RVV or emulate instructions.

Machine/compiler/source identities and actual results are retained in job logs.
No Torch headers, Torch installation, vLLM build, model evaluation, OpenMP worker,
performance result or independent physical attestation is claimed by this job.
It adds no GitHub App permission: the specifically authorized personal App is
already restricted to `luke-opus/vllm`. Both workflows are fork/branch limited and
use read-only repository permissions.
