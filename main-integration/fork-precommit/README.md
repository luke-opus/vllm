# Full manual pre-commit on hosted Linux

[GitHub run 41283200026](https://github.com/luke-opus/vllm/actions/runs/41283200026)
completed successfully. Each job executed the original command against its
independent pinned source checkout and unmodified configuration:

```sh
uvx pre-commit run --show-diff-on-failure --color=always --all-files --hook-stage manual
```

| Source | Exact commit | Manual hooks | Actual exit | Final Git diff/status |
| --- | --- | --- | --- | --- |
| base | `e0d00e4b8eb0c89ec30921d0695cfdd409e9c211` | 33 passed | 0 | empty |
| erf | `ec0fbf5e9b3e9d796ccae667a437f2e97c3f9cde` | 33 passed | 0 | empty |
| bf16 | `6592612016059859dff1e93bd534edc1a5737615` | 33 passed | 0 | empty |
| openmp | `dfdc85947fe914d04906f5d0206c020c208bd400` | 33 passed | 0 | empty |

The workflow commit is `90a7bbc95260d2c889880e636e79d145ce5e8e4e`;
it is not substituted for the actual tested source identities. See
[summary.json](summary.json) for each source/tree, job URL, artifact identity,
configuration hash, exact hook results and environment. Source identity gates
passed before execution. Before/after config hashes match original Git objects.
All downloaded ZIP hashes match GitHub artifact digests, archive CRCs passed,
and extracted bytes match ZIP entries. Root independently repeated these checks
and verified all actual exits, 33 result lines and empty source-status files.

The four environments report Ubuntu 24.04.5 x86-64, runner image
20261026.354.1, uv 0.13.6, pre-commit 4.7.0, Python discovery
`/usr/bin/python3.12`, and Docker 28.0.4. Docker was available for the original
graph hook. No SKIP, altered configuration, reduced file scope or source
write-back was used. The original manual stage excludes local Rust/commit-msg
hooks by its own configuration; separate local/DCO evidence remains valid.

Original files are kept under `artifacts/` and `jobs/`. The job-log download
first refused ANSI escape sequences; that collection error is recorded in
[collection.json](collection.json), with original stdout retained. The documented
`gh api --allow-escape-sequences` option was then used solely to save raw logs
to files. It did not rerun any job or change a CI result. Downloaded binary ZIPs
remain local; their verified digests are retained in the published metadata.

This is supplemental fork execution on GitHub-hosted Linux. It is not a pass
of the upstream self-hosted job, upstream eligibility/RTD gate, physical RISC-V
tests or human review. The upstream checks were not edited or overridden.
