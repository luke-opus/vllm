Current-main complete CPU extension build

Pinned upstream: e0d00e4b8eb0c89ec30921d0695cfdd409e9c211.
The base, ERF-only and BF16-only archives are independent source inputs. Their
archive identities and candidate trees are recorded in config-{variant}.json
and the parent sources/*.tar.manifest.json files. No combined-patch DSO is used.

This is QEMU RISC-V Linux execution, not native RISC-V hardware or upstream CI.
Native CMake uses the real main-venv Python, Torch, Clang 18, and oneDNN 3.13.
The fresh database has 231 units: 21 production units and 210 oneDNN units.
Every production unit is recompiled independently for each variant, at most two
compiler processes at a time. Production objects are cross-compiled on the Mac
with the real target sysroot/Torch headers, then linked inside RISC-V Linux.

oneDNN is reused only after guest-cache-audit.py compares all 210 generated
commands (normalizing relocated path prefixes only), generated oneDNN headers,
old/new Torch headers, and the previously recorded static-library SHA256.
Host-mirror and guest Torch headers are independently compared as well.
The 210 oneDNN units are not claimed as freshly recompiled in this run.

Disclosed environment adaptations, shared by all three variants:
- The already-built oneDNN contains the prior std::isfinite declaration include
  used for rvv_softmax.cpp; compat-math.hpp is retained here for full rebuilds.
- fla.cpp is compiled with -UVLLM_HAS_OPENBLAS, selecting the repository's real
  PyTorch reference BLAS fallback. No backend replacement or stub is supplied.
- The native generated link command receives real LLVM OpenMP link/rpath flags.
  Torch's real GNU OpenMP and the extension's LLVM OpenMP mappings are retained.
- Transformers is genuinely upgraded to 5.22.0 in the isolated main-venv;
  copied console-script wrappers are not used. Use its explicit Python path.

Actual build sequence per variant (base, erf, bf16):
1. guest-configure.py verifies/extracts the corresponding source archive, runs
   fresh CMake and captures the actual Ninja commands/database/generated headers.
2. guest-cache-audit.py verifies the dependency reuse conditions.
3. Transfer configure-{variant}.tar.gz to the host and extract into mirror/.
4. From the host workspace root:
   work/.venv/bin/python work/main-integration/build/replay.py \
     --config work/main-integration/build/config-{variant}.json --production-only
5. From the host workspace root:
   work/.venv/bin/python work/main-integration/build/package.py \
     work/main-integration/build/compile-{variant}.json
6. Transfer objects-{variant}.tar.gz and objects-{variant}.json to the guest.
7. main-venv/bin/python main-integration-build/guest-link.py {variant} --reuse-onednn
8. main-venv/bin/python main-integration-build/guest-load-check.py {variant}
The guest scripts reside under ~/vllm-validation/main-integration-build/;
main-venv/bin/python is relative to ~/vllm-validation/ in the invocation above.

Evidence locations:
- compile-{variant}.json: exact original/cross command, source and object SHA,
  adaptation, actual compiler exit, timeout flag, elapsed time, object identity.
- logs-{variant}/ and replay-{variant}-session.txt: raw compiler output/progress.
- objects-{variant}.json: verified archive/object/database/record identities.
- mirror/results/{variant}/: raw CMake/link logs, actual commands, dependency
  audit, final library SHA and actual mappings/outputs from the load check.
- link-load-{variant}-session.txt: complete native link/load output.
- observed-session-exits.json: actual observed coordinator session exits.

The direct load check calls three real current-main CPU operators and checks
their exact output. It is a build/load smoke check, not repository-test, model,
accuracy, performance, physical-hardware, or upstream-review evidence.
Repository/model/worker validation is performed separately after all builds.


Observed final outcomes
All 63 production compilations, three fresh configurations, three dependency
audits, three complete native links, and three load/operator-check drivers
completed with actual exit 0. No production compiler or native link timed out.
See observed-session-exits.json for the observed session completions.

base:
  tree/ref: e0d00e4b8eb0c89ec30921d0695cfdd409e9c211
  source archive SHA256: 4e29a49187b1b2b04f5b064fc39f4ef4fd9161898db692a88b7cf2e9d7cb69c6
  DSO SHA256: 57f4cc187d2c6502f176333331f7ba604926aa360710872478eae021d829b444
  DSO bytes: 215412368
  DSO path: /home/vllmtest/vllm-validation/main-integration-build/build/base/_C.abi3.so

erf:
  tree/ref: 94453352d881e58ebf423b67d7ccea2ba449eef5
  source archive SHA256: bb718659a9cb31de4c550b3c56d93c9021cef5a80eeab928e5846902dc87c804
  DSO SHA256: a1056a310cbfc357dfdcb18ff250be2fd8acf1d4bbb463682c6e6c492c341538
  DSO bytes: 215411528
  DSO path: /home/vllmtest/vllm-validation/main-integration-build/build/erf/_C.abi3.so

bf16:
  tree/ref: 8b860fc3eba99f6eccfe4b473c00a829b7ea9af3
  source archive SHA256: a0e956d990d50a69928d2db91a11155b5ff427db1c2ddc6c67438792d894dd96
  DSO SHA256: 93c8debb5754a1ba2241d97d9adf596274a22d8c4bf7a56ed755dc8d23bf6df2
  DSO bytes: 215431496
  DSO path: /home/vllmtest/vllm-validation/main-integration-build/build/bf16/_C.abi3.so

Executed script SHA256 (host originals were not changed after execution):
  guest-cache-audit.py: 00785d6eb68fc004e7f913754d23f834c047a07e418fc55ce241c91fe6560d46
  guest-configure.py: f19a542f77ccce2c8f1ed651e8da4fc5c073f5a9ac01d0b41b809ee28c52f56b
  guest-link.py: 8904e4f1c8619001a422502b9063a19ccf27f87f061b4e03948f7349282bb9e8
  guest-load-check.py: 372318dda497076c0fa2f605f601d57261038c6425b8500985b5aa334aa3c4e5
  package.py: 228e5a50e8cdcf291a0a8dd634f60a0dbb00b062b62e7de5b54d7c671df0f0e3
  replay.py: fec45a5e887d616149c44c8307e768f2e0c97c72963a38697dd9bc326f2d66a5

The four executed guest-script byte hashes also equal the host originals;
see guest-script-sha256.txt. build-summary.json records post-run verification
of object packages, exact compile-record/database hashes, source-archive hashes,
source identities, matching link/load DSO hashes, and actual loaded DSO paths.

Warnings are retained verbatim. Each variant emitted the same 28 fla.cpp
unused-comma-operand warnings. CMake also retained its existing kineto warning
and the RISC-V BF16-extension warning: the selected ISA has Zvfh, not Zvfbfmin.
This build/load smoke check alone does not prove BF16 operator/model behavior.

Path and dependency reproduction:
Raw logs retain the original host and guest absolute paths to preserve their
bytes and hashes. On another system update config-*.json path_mappings, mirror,
sysroot, and compiler paths to the corresponding real inputs; use a genuine
RISC-V Torch installation and the matching target headers. No credentials are
needed or included in this evidence set. The original full-extension toolchain
and oneDNN compatibility provenance are in ../../full-extension/README.md and
its evidence directory; those prior results are dependency provenance only,
not the current-main production compilation or execution results.

The separate current-main repository/model/worker validation begins only after
these three successful complete builds. No source worktree or PR was changed
by this build task. No upstream CI, native hardware, or review success is claimed.
