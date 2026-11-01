"""Record the driver exit and independently verify its reported restorations."""

import argparse
import json
import os
from pathlib import Path

from main_runtime_common import (
    HERE,
    PYTHON,
    RUNTIME,
    digest,
    install_signal_handlers,
    run_process,
    save,
)


def audit_states(restoration):
    checks = []
    for filename, expected in restoration.get("original_states", {}).items():
        path = Path(filename)
        assert path.absolute().is_relative_to(RUNTIME / "runs")
        if expected["kind"] == "absent":
            passed = not os.path.lexists(path)
        elif expected["kind"] == "link":
            passed = path.is_symlink() and os.readlink(path) == expected["target"]
        else:
            passed = (
                path.is_file()
                and not path.is_symlink()
                and digest(path) == expected["sha256"]
                and path.stat().st_mode == expected["mode"]
            )
        checks.append({"path": str(path), "expected": expected, "passed": passed})
    return {
        "checks": checks,
        "passed": bool(checks) and all(c["passed"] for c in checks),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--kind", choices=("repository", "bf16", "openmp"), required=True
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--case")
    parser.add_argument("--child-timeout", type=int, default=3600)
    parser.add_argument("--driver-timeout", type=int, default=14400)
    args = parser.parse_args()
    assert args.execute and HERE == RUNTIME
    assert args.run_id and all(c.isalnum() or c in "-_" for c in args.run_id)
    assert not args.case or args.kind == "repository"
    root = RUNTIME / "supervision" / args.run_id
    root.mkdir(parents=True, exist_ok=False)
    install_signal_handlers()
    command = [
        str(PYTHON),
        str(
            HERE
            / (
                "run-main-repository.py"
                if args.kind == "repository"
                else "run-main-model-pair.py"
            )
        ),
        "--execute",
        "--run-id",
        args.run_id,
        "--timeout",
        str(args.child_timeout),
    ]
    if args.kind != "repository":
        command += ["--kind", args.kind]
    if args.case:
        command += ["--case", args.case]
    report = {
        "driver_process": {},
        "independent_restoration_audit": [],
        "passed": False,
        "scope": "External exit/restoration observer; not assertions executed by the driver",
    }
    try:
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        run_process(
            command,
            HERE,
            env,
            root / "driver.txt",
            args.driver_timeout,
            report["driver_process"],
            termination_grace=45,
        )
    finally:
        result_path = (
            RUNTIME
            / "runs"
            / args.run_id
            / (
                "repository-results.json"
                if args.kind == "repository"
                else "pair-results.json"
            )
        )
        try:
            if result_path.exists():
                result = json.loads(result_path.read_text())
                report["driver_report_sha256"] = digest(result_path)
                report["driver_report_completed"] = result["completed"]
                for item in result.get("cases", result.get("variants", [])):
                    audit = audit_states(item.get("restoration", {}))
                    audit["name"] = item.get("name", item.get("variant"))
                    report["independent_restoration_audit"].append(audit)
                report["passed"] = (
                    report["driver_process"]["actual_returncode"] == 0
                    and not report["driver_process"]["timed_out"]
                    and not report["driver_process"]["interrupted"]
                    and result["completed"]
                    and bool(report["independent_restoration_audit"])
                    and all(
                        a["passed"] for a in report["independent_restoration_audit"]
                    )
                )
        except BaseException as error:
            report["audit_exception"] = repr(error)
            report["passed"] = False
            raise
        finally:
            save(root / "driver-exit-and-restoration.json", report)
    assert report["passed"], report


if __name__ == "__main__":
    main()
