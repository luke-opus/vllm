"""Prepared-only paired LLM checks on isolated copies of pinned main."""

import argparse
import hashlib
import json
from pathlib import Path

from main_runtime_common import (
    HERE,
    INPUTS,
    PYTHON,
    ROOT,
    digest,
    environment,
    replaced_files,
    run_process,
    save,
    stage_source,
    start_run,
    verified_libraries,
    verify_inputs,
    verify_production,
)


def python_source_inventory(source):
    files = {
        str(p.relative_to(source)): digest(p)
        for p in sorted((source / "vllm").rglob("*.py"))
    }
    serialized = json.dumps(files, sort_keys=True).encode()
    return {
        "count": len(files),
        "sha256": hashlib.sha256(serialized).hexdigest(),
        "files": files,
    }


def descriptive_comparison(base, fixed):
    assert base["parameters"] == fixed["parameters"]
    assert base["sampling_parameters"] == fixed["sampling_parameters"]
    assert len(base["outputs"]) == len(fixed["outputs"]) == 2
    differences = []
    comparisons = []
    for before, after in zip(base["outputs"], fixed["outputs"]):
        assert before["prompt"] == after["prompt"]
        assert before["prompt_token_ids"] == after["prompt_token_ids"]
        assert len(before["logprobs"]) == len(after["logprobs"]) == 4
        steps = []
        for index, (left, right) in enumerate(
            zip(before["logprobs"], after["logprobs"])
        ):
            same_prefix = before["token_ids"][:index] == after["token_ids"][:index]
            common = sorted(set(left) & set(right), key=int)
            deltas = {
                token: abs(left[token]["logprob"] - right[token]["logprob"])
                for token in common
            }
            if same_prefix:
                differences.extend(deltas.values())
            top_left = {
                token
                for token, value in left.items()
                if value["rank"] is not None and value["rank"] <= 5
            }
            top_right = {
                token
                for token, value in right.items()
                if value["rank"] is not None and value["rank"] <= 5
            }
            steps.append(
                {
                    "step": index,
                    "same_generated_prefix": same_prefix,
                    "common_returned_logprob_abs_differences": deltas,
                    "base_returned_token_count": len(left),
                    "fixed_returned_token_count": len(right),
                    "top5_set_equal": top_left == top_right,
                    "base_top5": sorted(top_left, key=int),
                    "fixed_top5": sorted(top_right, key=int),
                    "all_returned_token_set_equal": set(left) == set(right),
                }
            )
        comparisons.append(
            {
                "prompt": before["prompt"],
                "base_tokens": before["token_ids"],
                "fixed_tokens": after["token_ids"],
                "base_text": before["text"],
                "fixed_text": after["text"],
                "matching_tokens": sum(
                    a == b for a, b in zip(before["token_ids"], after["token_ids"])
                ),
                "steps": steps,
            }
        )
    return {
        "scope": (
            "Descriptive paired-run observations only. Aggregate deltas use only "
            "shared returned tokens at identical generated prefixes; no full-vocabulary, "
            "numerical equivalence, or model quality claim."
        ),
        "same_prefix_common_returned_logprob_values": len(differences),
        "same_prefix_max_abs_logprob_difference": max(differences)
        if differences
        else None,
        "same_prefix_mean_abs_logprob_difference": (
            sum(differences) / len(differences) if differences else None
        ),
        "steps_with_top5_set_changes": sum(
            not step["top5_set_equal"] for item in comparisons for step in item["steps"]
        ),
        "comparisons": comparisons,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--kind", choices=("bf16", "openmp"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()
    run = start_run(args.run_id, args.execute)
    report = {
        "base_ref": INPUTS["base_ref"],
        "kind": args.kind,
        "completed": False,
        "variants": [],
        "source_restored": False,
        "scope": "QEMU smoke coverage; prepared independently of the earlier-base runs",
    }
    source = None
    try:
        verify_inputs()
        libraries = verified_libraries()
        report["verified_libraries"] = libraries
        model_records = json.loads((HERE / "model-files.json").read_text())
        for record in model_records:
            path = ROOT / "pythia-14m" / record["name"]
            assert path.stat().st_size == record["bytes"]
            assert digest(path) == record["sha256"]
        report["model_files"] = model_records
        source = stage_source(run, "base", "source")
        original_python = python_source_inventory(source)
        report["original_python_source"] = original_python
        env = environment(source, run)
        if args.kind == "openmp":
            env.update(
                {
                    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                    "VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS": "60",
                }
            )
            before = (HERE / "inputs/base-cpu.py").read_bytes()
            after = (HERE / "inputs/openmp-cpu.py").read_bytes()
            old = b"                CpuArchEnum.S390X,\n            )\n"
            new = b"                CpuArchEnum.S390X,\n                CpuArchEnum.RISCV,\n            )\n"
            assert before.count(old) == 1 and after == before.replace(old, new, 1)
        report["controlled_environment"] = {
            name: value
            for name, value in env.items()
            if name
            in (
                "VLLM_TARGET_DEVICE",
                "VLLM_PLUGINS",
                "VLLM_ENABLE_V1_MULTIPROCESSING",
                "VLLM_CPU_CI_ENV",
                "VLLM_CPU_OMP_THREADS_BIND",
                "VLLM_NO_USAGE_STATS",
                "VLLM_WORKER_MULTIPROC_METHOD",
                "VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS",
                "OMP_NUM_THREADS",
                "GOMP_CPU_AFFINITY",
                "OMP_PLACES",
                "OMP_PROC_BIND",
                "KMP_AFFINITY",
                "HF_HOME",
                "HF_HUB_OFFLINE",
                "HF_HUB_DISABLE_TELEMETRY",
                "TORCHINDUCTOR_COMPILE_THREADS",
                "TORCHINDUCTOR_CACHE_DIR",
                "PYTHONPATH",
                "PYTHONDONTWRITEBYTECODE",
                "LD_PRELOAD",
                "CC",
                "CXX",
                "TMPDIR",
                "XDG_CACHE_HOME",
                "OPENBLAS_NUM_THREADS",
                "TOKENIZERS_PARALLELISM",
            )
        }
        report["complete_environment_sha256"] = hashlib.sha256(
            json.dumps(env, sort_keys=True).encode()
        ).hexdigest()
        report["default_runner_override_absent"] = "VLLM_USE_V2_MODEL_RUNNER" not in env
        for variant in ("base", "fixed"):
            library_variant = (
                "bf16" if args.kind == "bf16" and variant == "fixed" else "base"
            )
            library = libraries[library_variant]
            assert digest(library["path"]) == library["sha256"]
            assert python_source_inventory(source) == original_python
            changes = {source / "vllm/_C.abi3.so": Path(library["path"])}
            if args.kind == "openmp":
                filename = "base-cpu.py" if variant == "base" else "openmp-cpu.py"
                changes[source / "vllm/platforms/cpu.py"] = (
                    HERE / "inputs" / filename
                ).read_bytes()
            item = {
                "variant": variant,
                "library": library,
                "process": {},
                "restoration": {},
            }
            report["variants"].append(item)
            output = run / f"{variant}.json"
            command = [
                str(PYTHON),
                str(HERE / "check-main-model.py"),
                "--kind",
                args.kind,
                "--variant",
                variant,
                "--source",
                str(source),
                "--output",
                str(output),
                "--expected-library",
                library["path"],
                "--expected-sha256",
                library["sha256"],
            ]
            try:
                with replaced_files(changes, item["restoration"]):
                    item["cpu_source_sha256"] = digest(source / "vllm/platforms/cpu.py")
                    item["python_source_during_run"] = python_source_inventory(source)
                    run_process(
                        command,
                        source,
                        env,
                        run / f"{variant}.txt",
                        args.timeout,
                        item["process"],
                    )
                assert item["process"]["actual_returncode"] == 0
                assert (
                    not item["process"]["timed_out"]
                    and not item["process"]["interrupted"]
                )
                assert item["restoration"]["restored"]
                assert digest(library["path"]) == library["sha256"]
                assert python_source_inventory(source) == original_python
                result = json.loads(output.read_text())
                assert result["shutdown_returned"] is True
                assert result["worker"]["extension_sha256"] == library["sha256"]
                item["result_sha256"] = digest(output)
                item["completed"] = True
            finally:
                save(run / "pair-results.json", report)
        base, fixed = [
            json.loads((run / f"{variant}.json").read_text())
            for variant in ("base", "fixed")
        ]
        assert base["parameters"] == fixed["parameters"]
        assert base["sampling_parameters"] == fixed["sampling_parameters"]
        assert (
            base["worker"]["model_runner_class"]
            == fixed["worker"]["model_runner_class"]
        )
        assert base["worker"]["affinity"] == fixed["worker"]["affinity"]
        assert base["worker"]["torch_file"] == fixed["worker"]["torch_file"]
        assert base["vllm_distribution_metadata"] == fixed["vllm_distribution_metadata"]
        if args.kind == "openmp":
            assert (
                base["worker"]["extension_sha256"]
                == fixed["worker"]["extension_sha256"]
            )
            assert base["outputs"] == fixed["outputs"]
            report["exact_output_comparison"] = "equal: all returned output fields"
        else:
            assert (
                base["worker"]["extension_sha256"]
                != fixed["worker"]["extension_sha256"]
            )
            assert (
                report["variants"][0]["python_source_during_run"]
                == report["variants"][1]["python_source_during_run"]
            )
            comparison = descriptive_comparison(base, fixed)
            save(run / "comparison-details.json", comparison)
            report["descriptive_statistics"] = {
                k: v for k, v in comparison.items() if k != "comparisons"
            }
        report["completed"] = True
    except BaseException as error:
        report["exception"] = repr(error)
        raise
    finally:
        if source is not None:
            try:
                verify_production(source, "base")
                assert not (source / "vllm/_C.abi3.so").exists()
                assert not (source / "vllm/_C.abi3.so").is_symlink()
                report["source_restored"] = True
            except BaseException as error:
                report["restoration_exception"] = repr(error)
                report["completed"] = False
                save(run / "pair-results.json", report)
                raise
        save(run / "pair-results.json", report)


if __name__ == "__main__":
    main()
