# Official Ubuntu 26.04 runner route

The installed RISE App explicitly lists `ubuntu-26.04-riscv`; its current public
control-plane source maps that exact label set globally to SpacemiT K3. This
branch checks that separately from the existing Ubuntu24/rva23 queued job, which
is left intact. It does not assume that source configuration proves deployment
or available capacity. No new App or permission is needed.

The original production-header driver, input refs, test counts and assertions
remain unchanged. The workflow expects the actual Ubuntu26.04 environment and
uses its real default GCC/Clang packages instead of Ubuntu24-specific package
names. Compiler versions are recorded; no version metadata is rewritten. It
requires actual VLEN256/Zvfh execution and544 fixed ERF checks, plus the same
1458753 baseline/fixed BF16 conversions. Failures remain failures and a native
ERF base pass remains a valid observation rather than a manufactured failure.

This is header coverage, not full vLLM/operator/model/OpenMP or performance
evidence. Only an allocated, completed job with inspected raw outputs supplies
a result. The workflow is limited to the user fork and this independent branch.
