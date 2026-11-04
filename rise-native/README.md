# RISE K3 validation for the three RISC-V drafts

This branch reuses the published, unchanged production-header drivers in
`main-integration/native-plan/` to compare the exact base with ERF and BF16 PR
heads. It does not alter any source PR or upstream trust gate. The active
workflow is `.github/workflows/rise-native-validation.yml`, limited to
`luke-opus/vllm` and `riscv-rise-native-validation`.

## Resource selection

The RISE personal-account App must be installed on **only luke-opus/vllm**.
The actual installation page requests Actions and Metadata read, and repository
Administration read/write to register ephemeral runners. Installation requires the user's explicit action-time
confirmation; the existing Cloud-V grant does not authorize a different App.

The exact label set is `[ubuntu-24.04-riscv, rva23]`. The RISE control-plane
source inspected on 2026-11-04 provides a global route from that label set to
SpacemiT K3 from provider RISE. Its ordinary label alone selects Scaleway EM-RV1,
which is not suitable evidence for this RVV 1.0 path. V100 routing is restricted
to named entities and is not used here. Public source configuration is not
proof of deployment, available capacity, or allocation to this repository.

Sources:
- https://github.com/riseproject-dev/riscv-runner/blob/main/control-plane/internal/constants.go
- https://riscv-runners.riseproject.dev/docs/getting-started/install
- https://github.com/apps/rise-risc-v-runners-personal

## Evidence scope

The unchanged driver records the execution environment, original source blob
IDs and SHA256 hashes, native compiler commands, actual exit codes and raw
base/fixed outputs. BF16 must reproduce 667067 mismatches out of 1458753 base
checks and zero fixed mismatches. ERF must execute RVV VLEN256 and Zvfh, and
pass 544 fixed checks. A hardware baseline may retain agnostic lanes and pass;
that result must be reported without manufacturing a failure.

This only verifies production headers. It does not establish a complete vLLM
build, operator/model result, OpenMP worker behavior, performance, model quality,
upstream CI, or independent physical-silicon attestation. The provider advertises
native hardware; actual runner identity and execution still need inspection.

The job uses read-only repository permissions and bash steps, fetches its own
payload by exact GITHUB_SHA, and uses the pinned input commits and real Torch
headers already specified in the original plan. All failures remain visible.
No SKIP, emulation fallback, relaxed assertion or source mutation is added.

No run is claimed until a real job has been allocated and completed.
