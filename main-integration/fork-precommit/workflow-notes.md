# Supplemental Linux pre-commit execution

This branch runs the existing upstream command on four independent exact
snapshots: pinned main `e0d00e4b`, ERF `ec0fbf5e`, BF16 `65926120`, and OpenMP
`dfdc8594`. Each checkout verifies its full commit and tree identities before
running the unmodified `.pre-commit-config.yaml`.

```sh
uvx pre-commit run --show-diff-on-failure --color=always --all-files --hook-stage manual
```

This extends the earlier local changed-file checks to the full manual stage.
The upstream command and setup-uv/checkout action revisions are retained. The
runner is a standard GitHub-hosted Ubuntu 24.04 x86-64 machine, not the upstream
self-hosted runner and not RISC-V hardware. Runtime image and tool versions are
recorded. No hook is disabled and the repository's stage selection is unchanged.
Docker is required to be available so the graph-generation hook is not silently
avoided. The baseline runs the same command to distinguish existing whole-tree
or dependency-resolution failures from patch-specific failures.

Only this separate branch activates the workflow. It has read-only repository
permissions, no persistent checkout credential, and no source write-back.
Existing upstream PR workflows, eligibility checks, labels, branches and source
commits are unchanged. A successful supplemental job must never be reported as
passing upstream pre-commit, RTD, or the upstream trust requirement.

Raw command output, actual pre-commit/log-capture exits, tracked-file changes,
configuration hashes and source/runner identity are retained in each job's
artifact, including on failure. GitHub job status and logs establish whether a
run completed; a prepared workflow is not a result. No automatic formatting or
lockfile changes from these disposable checkouts are pushed into a PR.

All-files includes the original dependency-lock, full mypy, Docker graph and
repository-wide consistency hooks. If a base/fixed pair shares a failure, retain
both observations and diagnose it without weakening or omitting the hook. The
original configuration intentionally puts Rust formatting hooks in the local
pre-commit stage and sign-off in commit-msg; those are not selected by the
official manual-stage command. Actual source-commit DCO and local hooks remain
documented in the earlier immutable evidence.

AI assistance: OpenAI Codex. This branch supplies supplementary evidence only;
it is not a new upstream contribution or an assertion of human review.
