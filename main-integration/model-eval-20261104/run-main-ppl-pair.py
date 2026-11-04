"""Run isolated baseline/fixed BF16 scores and a separate FP32 reference."""

import argparse
import hashlib
import importlib.util
import json
import math
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--corpus-sha256", required=True)
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()
    run = start_run(args.run_id, args.execute)
    assert digest(args.corpus) == args.corpus_sha256
    spec = importlib.util.spec_from_file_location(
        "existing_model_pair", HERE / "run-main-model-pair.py"
    )
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    report = {
        "completed": False,
        "base_ref": INPUTS["base_ref"],
        "corpus_sha256": args.corpus_sha256,
        "variants": [],
        "source_restored": False,
        "scope": "QEMU-only finite-subset language-model evaluation; no native or general model quality claim",
    }
    source = None
    try:
        verify_inputs()
        libraries = verified_libraries()
        for record in json.loads((HERE / "model-files.json").read_text()):
            path = ROOT / "pythia-14m" / record["name"]
            assert (
                path.stat().st_size == record["bytes"]
                and digest(path) == record["sha256"]
            )
        source = stage_source(run, "base", "source")
        original = previous.python_source_inventory(source)
        report["python_source_inventory"] = original
        report["drivers"] = {
            name: digest(HERE / name)
            for name in [
                "check-main-ppl.py",
                "run-main-ppl-pair.py",
                "main_runtime_common.py",
                "check-main-model.py",
                "run-main-model-pair.py",
            ]
        }
        env = environment(source, run)
        report["environment_sha256"] = hashlib.sha256(
            json.dumps(env, sort_keys=True).encode()
        ).hexdigest()
        for variant in ("base", "fixed", "reference-fp32"):
            library = libraries["bf16" if variant == "fixed" else "base"]
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
                str(HERE / "check-main-ppl.py"),
                "--source",
                str(source),
                "--corpus",
                str(args.corpus),
                "--corpus-sha256",
                args.corpus_sha256,
                "--output",
                str(output),
                "--variant",
                variant,
                "--expected-library",
                library["path"],
                "--expected-sha256",
                library["sha256"],
            ]
            print(json.dumps({"event": "starting", "variant": variant}), flush=True)
            try:
                assert previous.python_source_inventory(source) == original
                with replaced_files(
                    {source / "vllm/_C.abi3.so": Path(library["path"])},
                    item["restoration"],
                ):
                    run_process(
                        command,
                        source,
                        env,
                        run / f"{variant}.txt",
                        args.timeout,
                        item["process"],
                    )
                assert item["restoration"]["restored"]
                assert digest(library["path"]) == library["sha256"]
                assert previous.python_source_inventory(source) == original
                assert item["process"]["actual_returncode"] == 0
                assert (
                    not item["process"]["timed_out"]
                    and not item["process"]["interrupted"]
                )
                result = json.loads(output.read_text())
                assert (
                    result["shutdown_returned"]
                    and result["metrics"]["scored_tokens"] == 4032
                )
                assert result["corpus_sha256"] == args.corpus_sha256
                assert result["worker"]["extension_sha256"] == library["sha256"]
                item.update(
                    {
                        "completed": True,
                        "result_sha256": digest(output),
                        "metrics": result["metrics"],
                    }
                )
            finally:
                save(run / "pair-results.json", report)
            print(
                json.dumps(
                    {
                        "event": "finished",
                        "variant": variant,
                        "actual_returncode": item["process"]["actual_returncode"],
                    }
                ),
                flush=True,
            )
        results = {
            v: json.loads((run / f"{v}.json").read_text())
            for v in ("base", "fixed", "reference-fp32")
        }
        assert results["base"]["parameters"] == results["fixed"]["parameters"]
        assert (
            results["base"]["sampling_parameters"]
            == results["fixed"]["sampling_parameters"]
        )
        reference = results["reference-fp32"]
        assert reference["parameters"] == {
            **results["base"]["parameters"],
            "dtype": "float32",
        }
        assert (
            reference["sampling_parameters"] == results["base"]["sampling_parameters"]
        )
        for value in results.values():
            assert value["worker"]["affinity"] == results["base"]["worker"]["affinity"]
            assert (
                value["worker"]["torch_file"] == results["base"]["worker"]["torch_file"]
            )
            assert (
                value["vllm_distribution_metadata"]
                == results["base"]["vllm_distribution_metadata"]
            )
        differences = {}
        for variant in ("base", "fixed"):
            deltas = []
            for row, expected in zip(results[variant]["samples"], reference["samples"]):
                assert (
                    row["row"] == expected["row"]
                    and row["token_ids_sha256"] == expected["token_ids_sha256"]
                )
                assert (
                    len(row["target_logprobs"])
                    == len(expected["target_logprobs"])
                    == 63
                )
                deltas.extend(
                    a - b
                    for a, b in zip(row["target_logprobs"], expected["target_logprobs"])
                )
            assert len(deltas) == 4032
            differences[variant] = {
                "target_logprob_mae_vs_fp32": math.fsum(abs(x) for x in deltas)
                / len(deltas),
                "target_logprob_rmse_vs_fp32": math.sqrt(
                    math.fsum(x * x for x in deltas) / len(deltas)
                ),
                "target_logprob_max_abs_delta_vs_fp32": max(abs(x) for x in deltas),
            }
        report["descriptive_comparison"] = differences
        report["completed"] = True
    except BaseException as error:
        report["exception"] = repr(error)
        raise
    finally:
        if source is not None:
            verify_production(source, "base")
            assert not (source / "vllm/_C.abi3.so").exists()
            assert not (source / "vllm/_C.abi3.so").is_symlink()
            report["source_restored"] = True
        save(run / "pair-results.json", report)


if __name__ == "__main__":
    main()
