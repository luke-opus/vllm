# Prepared follow-ups

These patches are reviewable local candidates, not commits in the upstream PR
heads. The original ERF and BF16 PRs remain drafts. No human-only attestation is
asserted by making these materials available.

* ERF: clarify model/operator approximation selection, `GeluAndMul`'s default
  `approximate="none"`, and automatic activation without a new RISC-V switch.
* BF16: add 18 exact production-header conversion cases and select them in the
  existing CPU hardware CI job; clarify RNE, canonical quiet NaNs, automatic
  software conversion and unchanged model dtype defaults.

The BF16 conversion test ran on host and real RISC-V Linux/QEMU GCC 13.3:
baseline 9 failed/9 passed; fixed 18 passed. See the root evidence directory.
The full original pre-commit hooks for the candidate source/tests and the
documentation clarifications passed.

Both complete candidate documentation builds passed strict checks with API
generation enabled. Live NumPy inventory downloads had failed; the successful
builds use a verified official NumPy 2.5 inventory cache from `numpy/doc`, Git
blob `63ee8d9cddacc54963d8622edeb499be38de0652`. Only the inventory transport URL
changed in temporary configs, preserving the official base URL and all checks.
The repository's `mkdocs.yaml` was unchanged. Local source findability passed
in a fresh agent; the candidate previews explicitly distinguish pending text
from the original PR heads.

Human review of every changed line and relevant tests is mandatory under the
upstream `AGENTS.md`. Model/public-entry validation and current-head upstream
CI are still outstanding; these materials do not waive them.
