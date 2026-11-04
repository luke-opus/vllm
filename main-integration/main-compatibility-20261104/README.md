# Newer-main patch applicability, 2026-11-04

The three complete draft patches apply to upstream main
`59519ec0ee0bc6b124c60fd50a82e94de98933e1` with actual check exit 0.
An isolated temporary Git index was loaded from that commit; the check used
`git apply --check --cached --whitespace=error` and each exact base-to-head diff.
The working trees and source PR heads were not changed.

This only checks applicability. Existing build, test, docs and model results
remain pinned to baseline `e0d00e4b8eb0c89ec30921d0695cfdd409e9c211`.
It is not a build/test result against this newer main and does not assert a
GitHub mergeability or CI result. Main may advance again after this check.

All three production fix files are unchanged upstream from the tested baseline.
The OpenMP PR's CI-selection/allowlist files have unrelated upstream changes;
the full patch still applies. The agent, test and PR-checklist instructions
were also compared and are unchanged. Blob identities and patch hashes are in
`summary.json`.

The first OpenMP check exited 128 because missing partial-clone blobs could not
be fetched under the restricted network. Its original diagnostic is retained
in `64080-apply.txt`; the separate network-available check exited 0 with an empty
diagnostic. This was an environment failure, not a patch-conflict result.

To reproduce in a clone with the recorded Git objects available, load a fresh
index from the recorded main, then check each included patch with the command
above after lossless decompression with `gzip -dc N.patch.gz`. The recorded
patch hashes cover the uncompressed original bytes. Do not reuse a user's staging index. Empty check logs represent actual
successful checks; they are included in the manifest.
