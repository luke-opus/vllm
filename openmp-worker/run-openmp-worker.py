"""Run the OpenMP base/fixed LLM comparison and restore the source afterward."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.home() / "vllm-validation")
    args = parser.parse_args()
    root = args.root.resolve()
    source = root / "model-source"
    path = source / "vllm/platforms/cpu.py"
    original = path.read_bytes()
    base = (root / "cpu-platform-openmp-base.py").read_bytes()
    fixed = (root / "cpu-platform-openmp-fixed.py").read_bytes()
    assert original == base, "The model source must start at the recorded base"
    old = b"                CpuArchEnum.S390X,\n            )\n"
    new = b"                CpuArchEnum.S390X,\n                CpuArchEnum.RISCV,\n            )\n"
    assert base.count(old) == 1 and fixed == base.replace(old, new, 1)
    library = root / "build-base/_C-bf16-linked-openmp.abi3.so"
    assert (source / "vllm/_C.abi3.so").resolve() == library.resolve()
    library_sha = digest(library)
    cpu = str(min(os.sched_getaffinity(0)))
    settings = {
        "CC": "/usr/bin/gcc-13",
        "CXX": "/usr/bin/g++-13",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "TORCHINDUCTOR_CACHE_DIR": str(root / "inductor-cache"),
        "VLLM_TARGET_DEVICE": "cpu",
        "VLLM_PLUGINS": "",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS": "60",
        "VLLM_CPU_OMP_THREADS_BIND": cpu,
        "VLLM_CPU_CI_ENV": "0",
        "VLLM_NO_USAGE_STATS": "1",
        "HF_HUB_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONPATH": str(source),
    }
    env = {**os.environ, **settings}
    for name in (
        "LD_PRELOAD",
        "GOMP_CPU_AFFINITY",
        "OMP_PLACES",
        "OMP_PROC_BIND",
        "KMP_AFFINITY",
    ):
        env.pop(name, None)
    records = []
    report_path = root / "openmp-worker-run-results.json"
    report = {
        "base_python_source": "ca8dbccf204abfcdf9dcb178f7e108c06b224b0f",
        "fixed_cpu_source": "018ef906c8cfc72cdfa21eafac157258c63373cb",
        "constant_cpp_ref": "174e56854703c0b9ede89e0a41f9e04dc4c59bc1",
        "constant_library_sha256": library_sha,
        "environment": settings,
        "variants": records,
        "exact_output_comparison": "not completed",
    }
    try:
        for variant, content in (("base", base), ("fixed", fixed)):
            assert digest(library) == library_sha
            path.write_bytes(content)
            log_path = root / f"openmp-worker-{variant}.txt"
            command = [
                str(root / ".venv/bin/python"),
                str(root / "check-openmp-worker.py"),
                variant,
                "--root",
                str(root),
            ]
            print(
                f"Running {variant}: real spawned CPUWorker and pinned model",
                flush=True,
            )
            timed_out = False
            with log_path.open("w") as log:
                proc = subprocess.Popen(
                    command,
                    cwd=root,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    returncode = proc.wait(timeout=3600)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
                    returncode = 124
                except BaseException:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    proc.wait()
                    raise
            if returncode != 0:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            records.append(
                {
                    "variant": variant,
                    "cpu_source_sha256": hashlib.sha256(content).hexdigest(),
                    "command": command,
                    "returncode": returncode,
                    "timed_out": timed_out,
                    "log": log_path.name,
                }
            )
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            print(
                f"{variant}: returncode={returncode}, log={log_path.name}", flush=True
            )
        assert all(item["returncode"] == 0 for item in records), records
        before = json.loads((root / "openmp-worker-base.json").read_text())
        after = json.loads((root / "openmp-worker-fixed.json").read_text())
        assert before["outputs"] == after["outputs"], "Base/fixed output mismatch"
        assert digest(library) == library_sha
        report["exact_output_comparison"] = (
            "passed: tokens, text and all returned logprobs are identical"
        )
    finally:
        path.write_bytes(original)
        report["source_restored"] = path.read_bytes() == original
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(
        "OpenMP base/fixed CPUWorker comparison passed; original source restored",
        flush=True,
    )


if __name__ == "__main__":
    main()
