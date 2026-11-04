"""Capture the PPL driver exit and audit restoration independently."""

import argparse
import importlib.util
import json
import os

from main_runtime_common import HERE, PYTHON, RUNTIME, digest, run_process, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--corpus-sha256", required=True)
    args = parser.parse_args()
    assert args.execute and HERE == RUNTIME
    assert args.run_id and all(c.isalnum() or c in "-_" for c in args.run_id)
    root = RUNTIME / "supervision" / args.run_id
    root.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location(
        "existing_restoration_auditor", HERE / "supervise-main.py"
    )
    auditor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auditor)
    report = {
        "driver_process": {},
        "independent_restoration_audit": [],
        "passed": False,
    }
    command = [
        str(PYTHON),
        str(HERE / "run-main-ppl-pair.py"),
        "--execute",
        "--run-id",
        args.run_id,
        "--corpus",
        args.corpus,
        "--corpus-sha256",
        args.corpus_sha256,
        "--timeout",
        "3600",
    ]
    try:
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        run_process(
            command,
            HERE,
            env,
            root / "driver.txt",
            14400,
            report["driver_process"],
            termination_grace=45,
        )
    finally:
        path = RUNTIME / "runs" / args.run_id / "pair-results.json"
        if path.exists():
            result = json.loads(path.read_text())
            report["driver_report_sha256"] = digest(path)
            report["driver_report_completed"] = result["completed"]
            for item in result["variants"]:
                check = auditor.audit_states(item["restoration"])
                check["variant"] = item["variant"]
                report["independent_restoration_audit"].append(check)
            report["passed"] = (
                report["driver_process"]["actual_returncode"] == 0
                and not report["driver_process"]["timed_out"]
                and not report["driver_process"]["interrupted"]
                and result["completed"]
                and len(report["independent_restoration_audit"]) == 3
                and all(x["passed"] for x in report["independent_restoration_audit"])
            )
        save(root / "driver-exit-and-restoration.json", report)
    print(json.dumps(report), flush=True)
    assert report["passed"], report


if __name__ == "__main__":
    main()
