"""Run selected unchanged repository tests through the real pinned-main root."""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from main_runtime_common import (
    HERE,
    INPUTS,
    PYTHON,
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

ACT = "tests/kernels/core/test_cpu_activation.py"
BOUNDARY = "tests/kernels/core/test_cpu_float_conversion.py"
PLATFORM = "tests/platforms/test_cpu.py"
CPU = "tests/v1/worker/cpu/"
STANDALONE = [
    CPU + "test_cpu_block_table.py",
    CPU + "test_cpu_input_batch.py",
    CPU
    + "test_cpu_gumbel_sample.py::test_greedy_supports_dtypes_strides_and_invalid_mapping",
    CPU
    + "test_cpu_gumbel_sample.py::test_pre_scaled_logits_match_temperature_application",
]


def cases():
    result = []
    for variant in ("base", "erf"):
        result.append(
            (
                f"erf-{variant}",
                variant,
                variant,
                {ACT: "erf-test_cpu_activation.py"},
                [ACT, "-k", "test_cpu_gelu_and_mul"],
                {"tests": 4, "failures": 2 if variant == "base" else 0},
            )
        )
    for variant in ("base", "bf16"):
        result.append(
            (
                f"bf16-operator-{variant}",
                variant,
                variant,
                {ACT: "bf16-test_cpu_activation.py"},
                [ACT, "-k", "rounds_to_nearest_even"],
                {"tests": 1, "failures": 1 if variant == "base" else 0},
            )
        )
        result.append(
            (
                f"bf16-boundary-{variant}",
                variant,
                variant,
                {BOUNDARY: "bf16-test_cpu_float_conversion.py"},
                [BOUNDARY],
                {"tests": 18, "failures": 9 if variant == "base" else 0},
            )
        )
    for variant in ("base", "openmp"):
        result.append(
            (
                f"openmp-config-{variant}",
                "base",
                "base",
                {
                    PLATFORM: "openmp-test_cpu.py",
                    "vllm/platforms/cpu.py": f"{variant}-cpu.py",
                },
                [PLATFORM, "-k", "riscv_preloads_bundled_openmp"],
                {"tests": 6, "failures": 2 if variant == "base" else 0},
            )
        )
    for variant in ("base", "erf", "bf16"):
        result.append(
            (f"standalone-{variant}", variant, variant, {}, STANDALONE, {"failures": 0})
        )
    return result


def read_junit(path):
    document = ET.parse(path).getroot()
    tests = list(document.iter("testcase"))
    failures = [
        dict(node.attrib, text=node.text or "")
        for case in tests
        for node in case.findall("failure")
    ]
    return {
        "tests": len(tests),
        "failures": len(failures),
        "errors": sum(len(case.findall("error")) for case in tests),
        "skipped": sum(len(case.findall("skipped")) for case in tests),
        "failure_details": failures,
        "nodeids": [case.attrib for case in tests],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--case", choices=[c[0] for c in cases()])
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    run = start_run(args.run_id, args.execute)
    report = {
        "base_ref": INPUTS["base_ref"],
        "completed": False,
        "cases": [],
        "scope": "Real root conftest; selected tests; explicit expected baseline regressions",
    }
    try:
        verify_inputs()
        libraries = verified_libraries()
        report["libraries"] = libraries
        for (
            name,
            source_variant,
            library_variant,
            replacements,
            targets,
            expected,
        ) in cases():
            if args.case and args.case != name:
                continue
            folder = run / name
            folder.mkdir()
            source = stage_source(folder, source_variant, "source")
            item = {
                "name": name,
                "source_variant": source_variant,
                "expected": expected,
                "restoration": {},
                "process": {},
            }
            report["cases"].append(item)
            changes = {
                source / rel: (HERE / "inputs" / filename).read_bytes()
                for rel, filename in replacements.items()
            }
            changes[source / "vllm/_C.abi3.so"] = Path(
                libraries[library_variant]["path"]
            )
            xml = folder / "pytest.xml"
            command = [
                str(PYTHON),
                "-m",
                "pytest",
                "-p",
                "main_import_probe",
                "-v",
                "-s",
                *targets,
                f"--junitxml={xml}",
                f"--basetemp={folder / 'pytest-tmp'}",
            ]
            try:
                with replaced_files(changes, item["restoration"]):
                    item["executed_test_sha256"] = {
                        rel: digest(source / rel)
                        for rel in replacements
                        if rel.startswith("tests/")
                    }
                    item["production_header_sha256"] = digest(
                        source / "csrc/cpu/float_convert.hpp"
                    )
                    item["root_conftest_sha256"] = digest(source / "tests/conftest.py")
                    env = environment(source, run)
                    env.update(
                        {
                            "PYTHONPATH": f"{source}:{HERE}",
                            "MAIN_VALIDATION_SOURCE": str(source),
                            "MAIN_VALIDATION_LIBRARY": libraries[library_variant][
                                "path"
                            ],
                            "MAIN_VALIDATION_LIBRARY_SHA256": libraries[
                                library_variant
                            ]["sha256"],
                            "MAIN_VALIDATION_IMPORT_RECORD": str(
                                folder / "actual-imports.json"
                            ),
                        }
                    )
                    run_process(
                        command,
                        source,
                        env,
                        folder / "pytest.txt",
                        args.timeout,
                        item["process"],
                    )
                verify_production(source, source_variant)
                stats = read_junit(xml)
                item["observed"] = stats
                item["actual_imports"] = json.loads(
                    (folder / "actual-imports.json").read_text()
                )
                assert (
                    item["actual_imports"]["start"]["extension_sha256"]
                    == libraries[library_variant]["sha256"]
                )
                assert (
                    item["actual_imports"]["finish"]["extension_sha256"]
                    == libraries[library_variant]["sha256"]
                )
                assert (
                    stats["tests"] > 0 and stats["errors"] == stats["skipped"] == 0
                ), stats
                for key, value in expected.items():
                    assert stats[key] == value, (name, key, stats[key], value)
                expected_exit = 1 if expected["failures"] else 0
                assert item["process"]["actual_returncode"] == expected_exit
                assert item["actual_imports"]["pytest_exit_status"] == expected_exit
                assert (
                    not item["process"]["timed_out"]
                    and not item["process"]["interrupted"]
                )
                for failure in stats["failure_details"]:
                    detail = json.dumps(failure)
                    assert "AssertionError" in detail or "assert " in detail, detail
                if name == "erf-base":
                    assert all(
                        "nan" in json.dumps(f).lower() for f in stats["failure_details"]
                    )
                if name == "bf16-operator-base":
                    assert "128 / 128" in json.dumps(stats["failure_details"])
                item["expected_behavior_confirmed"] = True
            finally:
                save(run / "repository-results.json", report)
        report["completed"] = True
    except BaseException as error:
        report["exception"] = repr(error)
        raise
    finally:
        save(run / "repository-results.json", report)


if __name__ == "__main__":
    main()
