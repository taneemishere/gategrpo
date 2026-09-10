from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .models import EvidencePacket, GateResult


def classify_failure(gates: list[GateResult]) -> str | None:
    for gate in gates:
        if not gate.passed:
            return gate.name
    return None


def build_evidence_packet(gates: list[GateResult]) -> EvidencePacket | None:
    for gate in gates:
        if gate.passed:
            continue
        fingerprint = gate.fingerprint or f"{gate.name}:generic"
        return EvidencePacket(
            evidence_id=_evidence_id(fingerprint),
            gate=gate.name,
            failure_type=gate.name,
            summary=_summarize_gate(gate),
            details=_bounded_details(gate),
            packet_ids=_packet_ids(gate),
        )
    return None


def context_packet_index(task_dir: Path) -> dict[str, dict[str, str]]:
    repo = task_dir / "repo"
    raw_config = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    allowed_paths = [str(path) for path in raw_config.get("allowed_paths", [])]
    candidates = [
        ("instructions", task_dir / "instructions.md"),
        ("spec", repo / "spec.md"),
    ]
    candidates.extend(("source", repo / relative) for relative in allowed_paths)
    packets: dict[str, dict[str, str]] = {}
    for kind, path in candidates:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        packet_id = _evidence_id(f"{kind}:{path.name}:{text}")
        packets[packet_id] = {"kind": kind, "path": str(path), "text": text}
    visible_dir = repo / "tests" / "visible"
    if visible_dir.exists():
        for path in sorted(visible_dir.glob("test_*.py")):
            text = path.read_text(encoding="utf-8")
            packet_id = _evidence_id(f"visible_test:{path.name}:{text}")
            packets[packet_id] = {"kind": "visible_test", "path": str(path), "text": text}
    return packets


def _summarize_gate(gate: GateResult) -> str:
    if gate.name == "scope_guard":
        blocked = gate.details.get("blocked_files", [])
        return f"patch touched blocked paths: {', '.join(blocked)}"
    if gate.name == "visible_tests":
        return "visible developer tests failed"
    if gate.name == "release_gate_regressions":
        return "release-gate regressions failed; failing test identifiers are provided"
    if gate.name == "python_ast":
        return "patched Python did not parse"
    if gate.name == "secret_scan":
        return "secret-like material detected"
    if gate.name == "patch_applies":
        return "candidate patch did not apply"
    return f"{gate.name} failed"


def _bounded_details(gate: GateResult) -> dict[str, object]:
    if gate.name == "release_gate_regressions":
        return {
            "returncode": gate.details.get("returncode"),
            "diagnostic": "regression gate failed; tracebacks and expected/actual values withheld — only the failing test identifiers are shown",
            "failing_tests": gate.details.get("failing_tests", []),
            "failing_test_count": gate.details.get("failing_test_count", 0),
        }
    if gate.name == "visible_tests":
        return {
            "returncode": gate.details.get("returncode"),
            "stdout_tail": str(gate.details.get("stdout", ""))[-1200:],
            "stderr_tail": str(gate.details.get("stderr", ""))[-1200:],
        }
    if gate.name == "scope_guard":
        return {"blocked_files": gate.details.get("blocked_files", [])}
    if gate.name == "python_ast":
        return {"failures": gate.details.get("failures", [])}
    if gate.name == "secret_scan":
        return {"findings": gate.details.get("findings", [])}
    return dict(gate.details)


def _packet_ids(gate: GateResult) -> tuple[str, ...]:
    if gate.name == "visible_tests":
        return ("visible-test-output", "task-spec", "source-under-test")
    if gate.name == "release_gate_regressions":
        return ("release-gate-limited-diagnostic", "task-spec", "source-under-test")
    if gate.name == "scope_guard":
        return ("scope-policy",)
    if gate.name == "python_ast":
        return ("source-under-test",)
    if gate.name == "secret_scan":
        return ("secret-policy",)
    return (f"{gate.name}-details",)


def _evidence_id(raw: str) -> str:
    digest = hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()[:12]
    return f"ev_{digest}"
