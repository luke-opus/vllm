"""Verify completed evaluation outputs and prepare an explicit public evidence set."""

import argparse
import gzip
import hashlib
import json
import math
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collected", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = read(args.collected / "pair-results.json")
    audit = read(args.collected / "driver-exit-and-restoration.json")
    selection = read(HERE / "corpus-selection.json")
    assert report["completed"] and report["source_restored"]
    assert audit["passed"] and audit["driver_report_completed"]
    assert audit["driver_process"]["actual_returncode"] == 0
    assert not audit["driver_process"]["timed_out"]
    assert not audit["driver_process"]["interrupted"]
    assert audit["driver_report_sha256"] == digest(args.collected / "pair-results.json")
    assert len(audit["independent_restoration_audit"]) == 3
    assert all(x["passed"] for x in audit["independent_restoration_audit"])
    assert report["corpus_sha256"] == selection["corpus_sha256"]
    results = {}
    metrics = {}
    expected_rows = [(x["row"], x["token_ids_sha256"]) for x in selection["samples"]]
    for item in report["variants"]:
        variant = item["variant"]
        assert variant not in results
        assert item["completed"] and item["restoration"]["restored"]
        assert item["process"]["actual_returncode"] == 0
        assert not item["process"]["timed_out"] and not item["process"]["interrupted"]
        path = args.collected / f"{variant}.json"
        assert digest(path) == item["result_sha256"]
        value = read(path)
        assert value["shutdown_returned"]
        assert value["corpus_sha256"] == selection["corpus_sha256"]
        assert value["worker"]["extension_sha256"] == item["library"]["sha256"]
        assert value["worker"]["use_v2_model_runner"] is True
        assert [
            (x["row"], x["token_ids_sha256"]) for x in value["samples"]
        ] == expected_rows
        assert all(
            len(x["target_logprobs"]) == len(x["target_ranks"]) == 63
            for x in value["samples"]
        )
        scores = [x for row in value["samples"] for x in row["target_logprobs"]]
        assert len(scores) == 4032 and all(math.isfinite(x) and x <= 0 for x in scores)
        nll = -math.fsum(scores) / len(scores)
        correct = sum(
            rank == 1 for row in value["samples"] for rank in row["target_ranks"]
        )
        expected = {
            "samples": 64,
            "scored_tokens": 4032,
            "mean_negative_log_likelihood": nll,
            "perplexity": math.exp(nll),
            "top1_correct": correct,
            "top1_accuracy": correct / 4032,
        }
        assert value["metrics"] == item["metrics"] == expected
        results[variant] = value
        metrics[variant] = expected
    assert set(results) == {"base", "fixed", "reference-fp32"}
    for variant in ("base", "fixed"):
        deltas = [
            a - b
            for row, ref in zip(
                results[variant]["samples"], results["reference-fp32"]["samples"]
            )
            for a, b in zip(row["target_logprobs"], ref["target_logprobs"])
        ]
        expected = {
            "target_logprob_mae_vs_fp32": math.fsum(abs(x) for x in deltas) / 4032,
            "target_logprob_rmse_vs_fp32": math.sqrt(
                math.fsum(x * x for x in deltas) / 4032
            ),
            "target_logprob_max_abs_delta_vs_fp32": max(abs(x) for x in deltas),
        }
        assert report["descriptive_comparison"][variant] == expected
    for name in ["check-main-ppl.py", "run-main-ppl-pair.py"]:
        assert digest(HERE / name) == report["drivers"][name]
    args.output.mkdir(parents=True, exist_ok=False)
    for name in [
        "PLAN.md",
        "prepare-corpus.py",
        "check-main-ppl.py",
        "run-main-ppl-pair.py",
        "supervise-main-ppl.py",
        "dataset-file.json",
        "corpus-selection.json",
        "tools-freeze.txt",
    ]:
        shutil.copyfile(HERE / name, args.output / name)
    for name in [
        "base.json",
        "fixed.json",
        "reference-fp32.json",
        "pair-results.json",
        "driver-exit-and-restoration.json",
    ]:
        shutil.copyfile(args.collected / name, args.output / name)
    logs = []
    for name in ["base.txt", "fixed.txt", "reference-fp32.txt", "driver.txt"]:
        original = (args.collected / name).read_bytes()
        target = args.output / (name + ".gz")
        target.write_bytes(gzip.compress(original, mtime=0))
        assert gzip.decompress(target.read_bytes()) == original
        logs.append(
            {
                "path": name + ".gz",
                "uncompressed_bytes": len(original),
                "uncompressed_sha256": hashlib.sha256(original).hexdigest(),
            }
        )
    summary = {
        "metrics": metrics,
        "differences_vs_fp32": report["descriptive_comparison"],
        "independent_numeric_recalculation": "passed",
        "raw_logs": logs,
        "scope": report["scope"],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    shutil.copyfile(HERE / "README-public.md", args.output / "README.md")
    shutil.copyfile(Path(__file__), args.output / Path(__file__).name)
    entries = [
        {"path": p.name, "bytes": p.stat().st_size, "sha256": digest(p)}
        for p in sorted(args.output.iterdir())
    ]
    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                "files": entries,
                "rule": "Explicit files only; raw corpus text/token inputs and dataset bytes are not redistributed. Original logs retained losslessly.",
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
