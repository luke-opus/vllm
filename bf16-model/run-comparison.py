"""Link matching OpenMP libraries, run real base/fixed models, restore source."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import tarfile
import time
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dynamic(path):
    return subprocess.check_output(["readelf", "-d", str(path)], text=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.home() / "vllm-validation")
    args = parser.parse_args()
    root = args.root.resolve()
    out = root / "bf16-model-comparison"
    source = root / "model-source"
    cpu_path = source / "vllm/platforms/cpu.py"
    original_cpu = cpu_path.read_bytes()
    assert original_cpu == (root / "cpu-platform-openmp-base.py").read_bytes()
    link = source / "vllm/_C.abi3.so"
    assert link.is_symlink()
    original_link = os.readlink(link)
    commands = json.loads((out / "native-link-commands.json").read_text())
    original = commands[-1]
    assert original.endswith(" && :")
    fixed = root / "build-base/_C-bf16-linked-openmp.abi3.so"
    base = out / "_C-base-linked-openmp.abi3.so"
    report = {
        "base_ref": "ca8dbccf204abfcdf9dcb178f7e108c06b224b0f",
        "fixed_ref": "174e56854703c0b9ede89e0a41f9e04dc4c59bc1",
        "constant_cpu_sha256": hashlib.sha256(original_cpu).hexdigest(),
        "variants": [],
        "comparison": "not completed",
    }
    report_path = out / "comparison.json"

    def save():
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    try:
        print(
            "Verifying pinned trained weights and existing fixed object inputs",
            flush=True,
        )
        for item in json.loads((out / "model-files.json").read_text()):
            assert digest(root / "pythia-14m" / item["name"]) == item["sha256"]
        with tarfile.open(root / "cross-bf16-reference-blas-objects.tar.gz") as bundle:
            for member in bundle:
                assert member.isfile() and member.name.startswith(
                    "build-base/CMakeFiles/"
                )
                assert (root / member.name).read_bytes() == bundle.extractfile(
                    member
                ).read()
        object_root = out / "base-objects"
        with tarfile.open(root / "cross-base-reference-blas-objects.tar.gz") as bundle:
            for member in bundle:
                assert member.isfile() and member.name.startswith(
                    "build-base/CMakeFiles/"
                )
                destination = object_root / member.name
                assert destination.resolve().is_relative_to(object_root.resolve())
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(bundle.extractfile(member).read())
        command = original.replace(" -o _C.abi3.so ", f" -o {base} ", 1)
        command = command.replace(
            " CMakeFiles/", f" {object_root}/build-base/CMakeFiles/"
        )
        command = command.removesuffix(" && :") + (
            " -L/usr/lib/llvm-18/lib -Wl,-rpath,/usr/lib/llvm-18/lib -lomp && :"
        )
        report["base_link_command"] = command
        report["common_onednn_sha256"] = digest(
            root / "build-base/_deps/onednn-build/src/libdnnl.a"
        )
        print("Linking complete base DSO with identical real LLVM OpenMP", flush=True)
        with (out / "link-base.txt").open("w") as log:
            subprocess.run(
                ["/bin/sh", "-c", command],
                cwd=root / "build-base",
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=300,
            )
        runtime = {}
        for variant, library in (("base", base), ("fixed", fixed)):
            text = dynamic(library)
            (out / f"dynamic-{variant}.txt").write_text(text)
            runtime[variant] = [
                line.split("]", 1)[0].split("[", 1)[1]
                for line in text.splitlines()
                if "(NEEDED)" in line or "(RUNPATH)" in line or "(RPATH)" in line
            ]
        assert runtime["base"] == runtime["fixed"], runtime
        assert "libomp.so.5" in runtime["base"]
        report["matching_dynamic_runtime"] = runtime
        settings = {
            "CC": "/usr/bin/gcc-13",
            "CXX": "/usr/bin/g++-13",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
            "TORCHINDUCTOR_CACHE_DIR": str(root / "inductor-cache"),
            "VLLM_TARGET_DEVICE": "cpu",
            "VLLM_PLUGINS": "",
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            "VLLM_CPU_OMP_THREADS_BIND": str(min(os.sched_getaffinity(0))),
            "VLLM_CPU_CI_ENV": "0",
            "VLLM_NO_USAGE_STATS": "1",
            "HF_HUB_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONPATH": str(source),
        }
        report["constant_environment"] = settings
        env = {**os.environ, **settings}
        for name in (
            "LD_PRELOAD",
            "GOMP_CPU_AFFINITY",
            "OMP_PLACES",
            "OMP_PROC_BIND",
            "KMP_AFFINITY",
        ):
            env.pop(name, None)
        for variant, library in (("base", base), ("fixed", fixed)):
            assert cpu_path.read_bytes() == original_cpu
            link.unlink()
            link.symlink_to(library)
            command = [
                str(root / ".venv/bin/python"),
                str(out / "check-model.py"),
                variant,
                "--root",
                str(root),
            ]
            print(
                f"Running {variant}: real LLM/Pythia-14m BF16, two prompts x four tokens",
                flush=True,
            )
            started = time.monotonic()
            with (out / f"{variant}.txt").open("w") as log:
                proc = subprocess.Popen(
                    command,
                    cwd=root,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    returncode = proc.wait(timeout=1200)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                    returncode = 124
                except BaseException:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                    raise
            record = {
                "variant": variant,
                "library": str(library),
                "library_sha256": digest(library),
                "command": command,
                "returncode": returncode,
                "seconds": time.monotonic() - started,
            }
            report["variants"].append(record)
            save()
            print(f"{variant} finished: returncode={returncode}", flush=True)
        assert all(item["returncode"] == 0 for item in report["variants"])
        before, after = [
            json.loads((out / f"{name}.json").read_text()) for name in ("base", "fixed")
        ]
        assert before["parameters"] == after["parameters"]
        for field in (
            "class",
            "device",
            "torch_threads",
            "affinity",
            "LD_PRELOAD",
            "loaded_openmp",
        ):
            assert before["worker"][field] == after["worker"][field], field
        report["comparison"] = {
            "tokens_equal": [
                a["token_ids"] == b["token_ids"]
                for a, b in zip(before["outputs"], after["outputs"])
            ],
            "all_returned_outputs_exactly_equal": before["outputs"] == after["outputs"],
            "base_tokens": [item["token_ids"] for item in before["outputs"]],
            "fixed_tokens": [item["token_ids"] for item in after["outputs"]],
            "note": "Observational comparison: no newly introduced numerical tolerance or claim that base must fail",
        }
    finally:
        if os.path.lexists(link):
            link.unlink()
        link.symlink_to(original_link)
        cpu_path.write_bytes(original_cpu)
        report["link_restored"] = os.readlink(link) == original_link
        report["cpu_source_restored"] = cpu_path.read_bytes() == original_cpu
        save()
    print("Comparison completed and source restored", flush=True)


if __name__ == "__main__":
    main()
