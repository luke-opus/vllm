"""Exercise the native scalar CPUWorker with original and fixed preload policies."""

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def inventory(source):
    return {
        str(p.relative_to(source)): digest(p) for p in (source / "vllm").rglob("*.py")
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-files", type=Path, required=True)
    parser.add_argument("--observer", type=Path, required=True)
    args = parser.parse_args()
    assert args.execute and os.uname().machine == "riscv64"
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.source.resolve()
    assert (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source, text=True
        ).strip()
        == "dfdc85947fe914d04906f5d0206c020c208bd400"
    )
    subprocess.run(
        [
            "git",
            "diff",
            "--exit-code",
            "e0d00e4b8eb0c89ec30921d0695cfdd409e9c211",
            "HEAD",
            "--",
            "csrc",
            "cmake",
        ],
        cwd=source,
        check=True,
    )
    import sys

    import torch
    import vllm._C
    from huggingface_hub import snapshot_download

    library = Path(vllm._C.__file__).resolve()
    library_sha = digest(library)
    model = args.output / "model"
    records = json.loads(args.model_files.read_text())
    snapshot_download(
        "EleutherAI/pythia-14m",
        revision="cf967c0a9a04383db6f7b1108d86b2962634b4ac",
        allow_patterns=[x["name"] for x in records],
        local_dir=model,
        token=False,
    )
    for record in records:
        path = model / record["name"]
        assert (
            path.stat().st_size == record["bytes"] and digest(path) == record["sha256"]
        )
    sandbox = args.output / "runtime-source"
    shutil.copytree(
        source,
        sandbox,
        symlinks=True,
        ignore=shutil.ignore_patterns(
            ".git", "build", "target", ".venv", "__pycache__", "_C.abi3.so"
        ),
    )
    (sandbox / "vllm/_C.abi3.so").symlink_to(library)
    policy = sandbox / "vllm/platforms/cpu.py"
    original = policy.read_bytes()
    before = subprocess.check_output(
        [
            "git",
            "show",
            "e0d00e4b8eb0c89ec30921d0695cfdd409e9c211:vllm/platforms/cpu.py",
        ],
        cwd=source,
    )
    after = subprocess.check_output(
        [
            "git",
            "show",
            "dfdc85947fe914d04906f5d0206c020c208bd400:vllm/platforms/cpu.py",
        ],
        cwd=source,
    )
    old = b"                CpuArchEnum.S390X,\n            )\n"
    new = b"                CpuArchEnum.S390X,\n                CpuArchEnum.RISCV,\n            )\n"
    assert before.count(old) == 1 and after == before.replace(old, new, 1)
    assert original == after
    original_inventory = inventory(sandbox)
    cpus = sorted(os.sched_getaffinity(0))[:2]
    assert len(cpus) == 2
    binding = ",".join(map(str, cpus))
    temporary = Path(tempfile.mkdtemp(prefix="vllm-native-omp-"))
    assert len(f"ipc://{temporary}/00000000-0000-0000-0000-000000000000".encode()) < 107
    env = os.environ.copy()
    for name in [
        "LD_PRELOAD",
        "GOMP_CPU_AFFINITY",
        "OMP_PLACES",
        "OMP_PROC_BIND",
        "KMP_AFFINITY",
        "VLLM_USE_V2_MODEL_RUNNER",
        "SETUPTOOLS_SCM_PRETEND_VERSION",
        "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM",
    ]:
        env.pop(name, None)
    env.update(
        {
            "PYTHONPATH": str(sandbox),
            "PYTHONDONTWRITEBYTECODE": "1",
            "VLLM_TARGET_DEVICE": "cpu",
            "VLLM_PLUGINS": "",
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            "VLLM_CPU_CI_ENV": "0",
            "VLLM_CPU_OMP_THREADS_BIND": binding,
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS": "60",
            "VLLM_NO_USAGE_STATS": "1",
            "HF_HUB_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "1",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
            "TMPDIR": str(temporary),
        }
    )
    report = {
        "completed": False,
        "source_restored": False,
        "build_ref": "dfdc85947fe914d04906f5d0206c020c208bd400",
        "library": str(library),
        "library_sha256": library_sha,
        "torch_file": torch.__file__,
        "torch_version": torch.__version__,
        "observer_sha256": digest(args.observer),
        "model_files": records,
        "affinity": cpus,
        "threads": 2,
        "cases": [],
        "scope": "Native-provider scalar CPU build and real spawned LLM worker; RVV/ERF, model quality, throughput and independent silicon attestation remain outside this run.",
    }
    try:
        for variant, data in [("base", before), ("fixed", after)]:
            policy.write_bytes(data)
            expected = dict(original_inventory)
            expected["vllm/platforms/cpu.py"] = hashlib.sha256(data).hexdigest()
            assert inventory(sandbox) == expected
            output = args.output / f"{variant}.json"
            command = [
                "taskset",
                "--cpu-list",
                binding,
                sys.executable,
                str(args.observer),
                "--kind",
                "openmp",
                "--variant",
                variant,
                "--source",
                str(sandbox),
                "--output",
                str(output),
                "--expected-library",
                str(library),
                "--expected-sha256",
                library_sha,
                "--model",
                str(model),
                "--threads",
                "2",
            ]
            item = {
                "variant": variant,
                "command": command,
                "actual_exit": None,
                "timed_out": False,
                "policy_sha256": digest(policy),
            }
            report["cases"].append(item)
            try:
                with (args.output / f"{variant}.txt").open("w") as stream:
                    proc = subprocess.Popen(
                        command,
                        cwd=sandbox,
                        env=env,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    item["pid"] = proc.pid
                    try:
                        proc.wait(timeout=1800)
                    except subprocess.TimeoutExpired:
                        item["timed_out"] = True
                        os.killpg(proc.pid, signal.SIGTERM)
                        try:
                            proc.wait(timeout=90)
                        except subprocess.TimeoutExpired:
                            os.killpg(proc.pid, signal.SIGKILL)
                            proc.wait()
                    item["actual_exit"] = proc.returncode
                assert item["actual_exit"] == 0 and not item["timed_out"]
                value = json.loads(output.read_text())
                assert (
                    value["shutdown_returned"] and value["worker"]["torch_threads"] == 2
                )
                # libgomp may pin the master thread to one selected CPU.
                assert value["worker"]["affinity"]
                assert set(value["worker"]["affinity"]).issubset(cpus)
                assert value["worker"]["extension_sha256"] == library_sha
                item["result_sha256"] = digest(output)
            finally:
                policy.write_bytes(original)
                assert inventory(sandbox) == original_inventory
                assert digest(library) == library_sha
                item["source_restored"] = True
                save(args.output / "pair-results.json", report)
        base = json.loads((args.output / "base.json").read_text())
        fixed = json.loads((args.output / "fixed.json").read_text())
        assert base["parameters"] == fixed["parameters"]
        assert base["sampling_parameters"] == fixed["sampling_parameters"]
        assert base["outputs"] == fixed["outputs"]
        assert base["worker"]["pid"] != fixed["worker"]["pid"]
        assert base["worker"]["torch_file"] == fixed["worker"]["torch_file"]
        assert base["vllm_distribution_metadata"] == fixed["vllm_distribution_metadata"]
        report["exact_output_comparison"] = "all returned output fields equal"
        report["completed"] = True
    except BaseException as error:
        report["exception"] = repr(error)
        raise
    finally:
        policy.write_bytes(original)
        assert inventory(sandbox) == original_inventory
        report["source_restored"] = True
        save(args.output / "pair-results.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
