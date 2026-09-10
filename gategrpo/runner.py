from __future__ import annotations

import shutil
import time
from pathlib import Path

from .gates import python_ast_gate, pytest_gate, scope_guard, secret_scan_gate
from .models import GateResult, RunResult
from .patching import apply_patch_file, check_patch_applies, parse_touched_files
from .task import load_task
from .tracing import TraceWriter


def run_task(task_dir: Path, run_dir: Path, patch_file: Path | None = None) -> RunResult:
    started = time.monotonic()
    task = load_task(task_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    trace_path = run_dir / "trace.jsonl"
    if trace_path.exists():
        trace_path.unlink()
    trace = TraceWriter(trace_path)
    workspace = run_dir / "workspace"
    if workspace.exists():
        shutil.rmtree(workspace)
    shutil.copytree(task.workspace, workspace)

    candidate_patch = patch_file or task.default_patch
    gates: list[GateResult] = []
    trace.emit("run_started", task_id=task.task_id, task_name=task.name, workspace=str(workspace))
    if candidate_patch is None:
        gate = GateResult(name="candidate_patch", passed=False, details={"error": "no_patch_provided"})
        gates.append(gate)
        trace.emit("gate_completed", name=gate.name, passed=gate.passed, details=gate.details)
        return _finish(task.task_id, "failed", workspace, trace_path, gates, trace, (), started)

    candidate_patch = candidate_patch.resolve()
    patch_text = candidate_patch.read_text(encoding="utf-8")
    touched_files = parse_touched_files(patch_text)
    trace.emit("candidate_loaded", patch_file=str(candidate_patch), touched_files=touched_files)

    gate = scope_guard(touched_files, task.allowed_paths)
    gates.append(gate)
    trace.emit("gate_completed", name=gate.name, passed=gate.passed, details=gate.details)
    if not gate.passed:
        return _finish(task.task_id, "rejected", workspace, trace_path, gates, trace, tuple(touched_files), started)

    check = check_patch_applies(workspace, candidate_patch)
    gate = GateResult(
        name="patch_applies",
        passed=check.passed,
        details={"stdout": check.stdout, "stderr": check.stderr, "returncode": check.returncode},
        fingerprint=None if check.passed else f"patch_applies:{check.stderr[-200:]}",
    )
    gates.append(gate)
    trace.emit("gate_completed", name=gate.name, passed=gate.passed, details=gate.details)
    if not gate.passed:
        return _finish(task.task_id, "rejected", workspace, trace_path, gates, trace, tuple(touched_files), started)

    applied = apply_patch_file(workspace, candidate_patch)
    trace.emit("patch_applied", passed=applied.passed, returncode=applied.returncode)
    if not applied.passed:
        gate = GateResult(
            name="patch_apply_runtime",
            passed=False,
            details={"stderr": applied.stderr},
            fingerprint=f"patch_apply_runtime:{applied.stderr[-200:]}",
        )
        gates.append(gate)
        trace.emit("gate_completed", name=gate.name, passed=gate.passed, details=gate.details)
        return _finish(task.task_id, "rejected", workspace, trace_path, gates, trace, tuple(touched_files), started)

    for gate in (
        python_ast_gate(workspace, touched_files),
        secret_scan_gate(workspace),
        pytest_gate(workspace, task.visible_tests, "visible_tests"),
        pytest_gate(workspace, task.regression_tests, "release_gate_regressions"),
    ):
        gates.append(gate)
        trace.emit("gate_completed", name=gate.name, passed=gate.passed, details=gate.details)
        if not gate.passed:
            return _finish(task.task_id, "rejected", workspace, trace_path, gates, trace, tuple(touched_files), started)

    return _finish(task.task_id, "promoted", workspace, trace_path, gates, trace, tuple(touched_files), started)


def _finish(
    task_id: str,
    status: str,
    workspace: Path,
    trace_path: Path,
    gates: list[GateResult],
    trace: TraceWriter,
    touched_files: tuple[str, ...],
    started: float,
) -> RunResult:
    duration_seconds = round(time.monotonic() - started, 6)
    trace.emit(
        "run_finished",
        task_id=task_id,
        status=status,
        gates=[{"name": gate.name, "passed": gate.passed} for gate in gates],
        touched_files=list(touched_files),
        duration_seconds=duration_seconds,
    )
    return RunResult(
        task_id=task_id,
        status=status,
        workspace=workspace,
        trace_path=trace_path,
        gates=gates,
        touched_files=touched_files,
        duration_seconds=duration_seconds,
    )
