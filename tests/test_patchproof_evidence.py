from patchproof.evidence import build_evidence_packet
from patchproof.models import GateResult


def test_release_gate_evidence_includes_failing_test_identifiers():
    gate = GateResult(
        name="release_gate_regressions",
        passed=False,
        details={
            "returncode": 1,
            "stdout": "withheld",
            "stderr": "withheld",
            "diagnostic": "release-gate output withheld from repair loop",
            "failing_tests": [
                "tests/regression/test_checkout.py::test_half_up_rounding_cases_are_correct",
            ],
            "failing_test_count": 1,
        },
    )

    packet = build_evidence_packet([gate])

    assert packet is not None
    assert packet.details["returncode"] == 1
    assert packet.details["failing_tests"] == [
        "tests/regression/test_checkout.py::test_half_up_rounding_cases_are_correct",
    ]
    assert packet.details["failing_test_count"] == 1
    assert "failing test identifiers" in packet.summary
    assert "stdout_tail" not in packet.details
    assert "stderr_tail" not in packet.details
