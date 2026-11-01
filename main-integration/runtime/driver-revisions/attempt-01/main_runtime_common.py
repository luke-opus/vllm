"""Shared, isolated execution plumbing for the pinned-main validation drivers."""

import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path("/home/vllmtest/vllm-validation")
RUNTIME = ROOT / "main-integration-runtime"
BUILD = ROOT / "main-integration-build"
PYTHON = ROOT / "main-venv/bin/python"
INPUTS = json.loads((HERE / "driver-inputs.json").read_text())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, data):
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


class Interrupted(RuntimeError):
    pass


def install_signal_handlers():
    def interrupt(signum, frame):
        raise Interrupted(f"Received signal {signum}; cleanup required")

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)


def start_run(name, execute):
    if not execute:
        raise SystemExit(
            "Prepared only. Root authorization and --execute are required."
        )
    assert HERE == RUNTIME, (HERE, RUNTIME)
    assert name and all(c.isalnum() or c in "-_" for c in name)
    run = RUNTIME / "runs" / name
    run.mkdir(parents=True, exist_ok=False)
    install_signal_handlers()
    return run


def verify_inputs():
    for name, record in INPUTS["inputs"].items():
        assert digest(HERE / "inputs" / name) == record["sha256"], name


def verified_libraries():
    result = {}
    for variant in ("base", "erf", "bf16"):
        records = BUILD / "results" / variant
        link = json.loads((records / "link.json").read_text())
        load = json.loads((records / "load-check.json").read_text())
        library = BUILD / "build" / variant / "_C.abi3.so"
        assert link["variant"] == load["variant"] == variant
        assert link["returncode"] == 0 and not link["timed_out"]
        sha = digest(library)
        assert sha == link["library_sha256"] == load["library_sha256"]
        assert Path(link["library"]) == library
        identity = INPUTS["sources"][variant]
        identity_field = "index_tree" if "index_tree" in identity else "commit"
        assert link["source_identity"][identity_field] == identity[identity_field]
        if "main_commit" in identity:
            assert link["source_identity"]["main_commit"] == INPUTS["base_ref"]
        assert link["source_identity"]["sha256"] == identity["sha256"]
        result[variant] = {
            "path": str(library),
            "sha256": sha,
            "source_identity": link["source_identity"],
        }
    return result


def stage_source(run, variant, label=None):
    source = BUILD / "source" / f"main-source-{variant}"
    destination = run / (label or variant)
    shutil.copytree(
        source,
        destination,
        symlinks=True,
        ignore=shutil.ignore_patterns("__pycache__", "_C.abi3.so"),
    )
    verify_production(destination, variant)
    return destination


def verify_production(source, variant):
    for relative, expected in INPUTS["production_files"][variant].items():
        assert digest(source / relative) == expected, relative


def environment(source, run):
    env = os.environ.copy()
    for name in (
        "LD_PRELOAD",
        "GOMP_CPU_AFFINITY",
        "OMP_PLACES",
        "OMP_PROC_BIND",
        "KMP_AFFINITY",
        "VLLM_USE_V2_MODEL_RUNNER",
        "VIRTUAL_ENV",
        "SETUPTOOLS_SCM_PRETEND_VERSION",
        "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM",
    ):
        env.pop(name, None)
    for directory in ("tmp", "cache", "hf-cache", "inductor-cache"):
        (run / directory).mkdir(exist_ok=True)
    env.update(
        {
            "PYTHONPATH": str(source),
            "PYTHONDONTWRITEBYTECODE": "1",
            "CC": "/usr/bin/gcc-13",
            "CXX": "/usr/bin/g++-13",
            "VLLM_TARGET_DEVICE": "cpu",
            "VLLM_PLUGINS": "",
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            "VLLM_CPU_CI_ENV": "0",
            "VLLM_CPU_OMP_THREADS_BIND": str(min(os.sched_getaffinity(0))),
            "VLLM_NO_USAGE_STATS": "1",
            "HF_HUB_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
            "TORCHINDUCTOR_CACHE_DIR": str(run / "inductor-cache"),
            "TMPDIR": str(run / "tmp"),
            "XDG_CACHE_HOME": str(run / "cache"),
            "HF_HOME": str(run / "hf-cache"),
        }
    )
    return env


@contextmanager
def replaced_files(changes, restoration):
    """Change only disposable runtime snapshots; restore bytes or original links."""
    original = {}
    restoration["original_states"] = {}
    try:
        for path, value in changes.items():
            path = Path(path)
            assert path.absolute().is_relative_to(RUNTIME / "runs")
            if path.is_symlink():
                original[path] = ("link", os.readlink(path))
            elif path.exists():
                original[path] = ("file", path.read_bytes(), path.stat().st_mode)
            else:
                original[path] = ("absent",)
            old = original[path]
            restoration["original_states"][str(path)] = (
                {"kind": "link", "target": old[1]}
                if old[0] == "link"
                else {
                    "kind": "file",
                    "sha256": hashlib.sha256(old[1]).hexdigest(),
                    "mode": old[2],
                }
                if old[0] == "file"
                else {"kind": "absent"}
            )
            if os.path.lexists(path):
                path.unlink()
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(value, Path):
                path.symlink_to(value)
            else:
                path.write_bytes(value)
        yield
    finally:
        errors = []
        for path, old in reversed(list(original.items())):
            try:
                if os.path.lexists(path):
                    path.unlink()
                if old[0] == "link":
                    path.symlink_to(old[1])
                    assert os.readlink(path) == old[1]
                elif old[0] == "file":
                    path.write_bytes(old[1])
                    path.chmod(old[2])
                    assert path.read_bytes() == old[1]
                else:
                    assert not os.path.lexists(path)
            except BaseException as error:
                errors.append(f"{path}: {error!r}")
        restoration.update(
            {
                "checked": True,
                "restored": not errors,
                "errors": errors,
                "paths": [str(p) for p in original],
            }
        )
        if errors:
            raise RuntimeError(errors)


def run_process(command, cwd, env, output, timeout, record, termination_grace=10):
    record.update(
        {
            "command": command,
            "cwd": str(cwd),
            "timed_out": False,
            "interrupted": False,
            "actual_returncode": None,
        }
    )
    started = time.monotonic()
    proc = None
    try:
        with output.open("w") as stream:
            proc = subprocess.Popen(
                command,
                cwd=cwd,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            record["pid"] = proc.pid
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                record["timed_out"] = True
                stop_process(proc, record, termination_grace)
            except BaseException:
                record["interrupted"] = True
                stop_process(proc, record, termination_grace)
                raise
    finally:
        if proc is not None:
            if proc.poll() is None:
                stop_process(proc, record, termination_grace)
            elif proc.returncode != 0:
                signal_process_group(proc, signal.SIGTERM, record)
            record["actual_returncode"] = proc.returncode
        record["seconds"] = time.monotonic() - started


def signal_process_group(proc, signum, record):
    try:
        os.killpg(proc.pid, signum)
        record.setdefault("group_signals_sent", []).append(signum)
    except ProcessLookupError:
        pass


def stop_process(proc, record, termination_grace):
    signal_process_group(proc, signal.SIGTERM, record)
    if proc.poll() is not None:
        return
    try:
        proc.wait(timeout=termination_grace)
    except subprocess.TimeoutExpired:
        signal_process_group(proc, signal.SIGKILL, record)
        proc.wait()
