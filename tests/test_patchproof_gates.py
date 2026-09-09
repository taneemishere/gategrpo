from pathlib import Path

from patchproof.runner import run_task


ROOT = Path(__file__).resolve().parents[1]
ORDER_TASK = ROOT / "tasks" / "order_service"


def test_release_gate_failure_reports_failing_test_identifiers(tmp_path):
    result = run_task(ORDER_TASK, tmp_path / "order_run", patch_file=ORDER_TASK / "candidate_regression_fail.patch")

    release_gate = next(gate for gate in result.gates if gate.name == "release_gate_regressions")

    assert not release_gate.passed
    assert release_gate.details["stdout"] == "withheld"
    assert release_gate.details["stderr"] == "withheld"
    assert release_gate.details["failing_test_count"] == len(release_gate.details["failing_tests"])
    assert release_gate.details["failing_tests"]
    assert all(" - " not in test_id and "assert" not in test_id for test_id in release_gate.details["failing_tests"])
