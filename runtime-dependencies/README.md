# RISC-V runtime dependency preparation

These records describe the validation environment, not changes in any upstream
PR. A successful package import does not establish vLLM/model/CI success.

## Harmony

The exact PyPI `oss-harmony==0.0.12` source archive has SHA256
`6a2dd2db3ecd5ac771768c0a3091a2e1ee1354cbfcd61efa1283c26f300f0fcd`.
Its native Rust build was progressing slowly under QEMU. A locked release
cross-build used task-local Rust 1.99.1, Maturin 1.16.0, cargo-zigbuild 0.24.0
and Zig 0.17.0, following the supported
[Maturin cross-compilation workflow](https://www.maturin.rs/distribution.html#cross-compiling).
The release compilation finished in 1m18s.

```sh
PYO3_CROSS=1 PYO3_CROSS_PYTHON_VERSION=3.12 \
PYO3_CROSS_PYTHON_IMPLEMENTATION=CPython \
maturin build --release --locked --zig \
  --target riscv64gc-unknown-linux-gnu \
  --compatibility manylinux_2_39 --out ../wheels
```

The real RISC-V ELF extension passed import and encoding/decoding round-trip
checks in an isolated target directory inside the guest. Only after that
verification was the incomplete native dependency build canceled and the same
version wheel installed into the validation venv. No stub module, lowered
dependency constraint or changed release optimization was used.

The common requirements then installed successfully; `uv pip check` reported
148 compatible packages. This does not claim that every CPU/test requirement,
including optional N-gram dependencies, was installed.

## XGrammar

The first actual repository pytest invocation stopped while importing the root
`conftest.py`/`LLM`: `ModuleNotFoundError: No module named 'xgrammar'`. This was
collection exit code 4, not the expected numerical regression failure.
The unchanged requirements exclude RISC-V from XGrammar installation, while
`vllm/parser/abstract_parser.py` imports its classes directly. The same import
is present in the inspected current main.

[RISE](https://pypi.riseproject.dev/simple/xgrammar/) provides real
`xgrammar==0.2.8` and compatible TVM FFI RISC-V wheels.
Direct guest GitHub downloads failed TLS negotiation. Host HTTP/1.1 downloads
were verified against the RISE index SHA256 values and transferred over the
task-local payload server. Both packages installed without replacing any vLLM
objects; all 152 installed packages passed the dependency check, and the native
`xgrammar.Grammar.from_regex("a+")` call passed.

The exact artifact URLs/checksums are in `extra-riscv-wheels.json`. Repository
pytest initially progressed to the next import gap described below.

## LLGuidance

Both the repository pytest root and the independent LLM script next stopped on
`llguidance`: the RISC-V requirements marker excludes it, but the
`GuidanceGrammar.ll_matcher` type annotation evaluates the lazy module. The
inspected main retains that marker and annotation. These were import/collection
failures, not the expected numerical before-fix failures.

RISE's available wheels stop at 1.5.0, below the required `>=1.7,<1.8` range.
The verified official PyPI `llguidance==1.7.6` source archive was cross-built
without modifying it, using its pinned Rust 1.95.0 and the installed RISC-V
standard library. The locked optimized release build completed in 4m38s.
Its native RISC-V module and `LLMatcher.grammar_from_regex("a+")` both passed;
all 153 installed packages passed `uv pip check`.

Repository pytest and the independent real-weight LLM run were then restarted
after fixing the diagnosed dependencies. They remain in progress; this record
does not assert their success. No tests, tolerance thresholds or vLLM backend
objects were replaced to bypass the imports.
