# Native RISC-V scalar BF16 validation, 2026-11-04

The assigned RISE runner completed the exact production-header comparison for
base `e0d00e4b8eb0c89ec30921d0695cfdd409e9c211` and BF16 draft head
`6592612016059859dff1e93bd534edc1a5737615`. The job did not invoke an emulator.
RISE advertises this standard label as native Scaleway EM-RV1 hardware; the job
records riscv64 Linux and ISA `rv64imafdcsu`. Physical silicon was not independently
attested, and this result does not certify RVV 1.0 or the ERF implementation.

[Actual workflow run](https://github.com/luke-opus/vllm/actions/runs/41830864789),
job 126577392408, runner ID 21, workflow head
`5beb1df43497ec58832522493c93286990f7b2cc`. All actual job steps succeeded.

| Variant | Conversions | Mismatches | Actual test exit |
| --- | ---: | ---: | ---: |
| Base | 1458753 | 667067 | 1 |
| Fixed | 1458753 | 0 | 0 |

The nonzero base test exit is expected and retained. The workflow checks this
exact base count and requires the fixed exit/count to be zero. It compiles the
original `float_convert.hpp` with native GCC 13.3.0, C++20, `-O2 -Wall -Wextra
-Werror`, without forcing vector instructions or changing production code. The
comparison uses the previously published independent nearest-representable-value
reference; it does not duplicate the fixed rounding formula as its oracle.

The raw job log (lossless gzip), job/run metadata, exact workflow and drivers
are retained here. Gzip preserves the original whitespace and bytes without
introducing formatting errors into the evidence diff. To inspect it, run
`gzip -dc scalar-job-126577392408.log.gz > scalar-job-126577392408.log`; verify
the uncompressed SHA256 against `scalar-result.json`.
The source blob IDs and SHA256 values printed by the job were independently
checked against the original local Git objects. The raw status rows confirm
base exit 1 and fixed exit 0. See `scalar-result.json` and `manifest.json`.

This adds native scalar-helper coverage. Full vLLM/operator/model execution,
OpenMP workers, RVV, ERF and performance are not established by this job. The
separate K3 job 41829702428 was still queued in the latest check; no result is
inferred from its requested labels. Upstream CI and human review remain separate
requirements. Existing source PRs and their unchecked validation items stay intact.
