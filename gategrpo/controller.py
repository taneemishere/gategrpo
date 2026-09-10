from __future__ import annotations

import time
from pathlib import Path

from .archive import CandidateArchive, build_run_metadata
from .critic import CleanContextCritic
from .evidence import build_evidence_packet, classify_failure
from .models import CandidateRecord, CandidateSpec, SearchBudget, SearchResult
from .runner import run_task
from .task import load_task
from .tracing import TraceWriter


def run_search(
    task_dir: Path,
    run_dir: Path,
    budget: SearchBudget | None = None,
    candidate_files: list[Path | CandidateSpec] | None = None,
    model: str | None = None,
    provider: str | None = None,
    temperature: float | None = None,
) -> SearchResult:
    task = load_task(task_dir)
    budget = budget or SearchBudget()
    started = time.monotonic()
    run_dir.mkdir(parents=True, exist_ok=True)
    trace_path = run_dir / "controller_trace.jsonl"
    if trace_path.exists():
        trace_path.unlink()
    trace = TraceWriter(trace_path)
    archive = CandidateArchive()
    critic = CleanContextCritic()
    raw_candidates = candidate_files or list(task.candidate_patches or (() if task.default_patch is None else (task.default_patch,)))
    candidates = [_candidate_spec(candidate) for candidate in raw_candidates]

    trace.emit(
        "search_started",
        task_id=task.task_id,
        max_candidates=budget.max_candidates,
        max_repeated_failures=budget.max_repeated_failures,
        candidate_count=len(candidates),
    )

    stop_reason = "candidate_budget_exhausted"
    remaining = list(candidates)
    evaluated = 0
    while remaining and evaluated < budget.max_candidates:
        if _wall_budget_exhausted(started, budget):
            stop_reason = "wall_clock_budget_exhausted"
            trace.emit("search_stopped", reason=stop_reason)
            break
        evaluated += 1
        candidate_id = f"candidate_{evaluated:03d}"
        route = "initial_repair" if not archive.records else critic.route_next(archive.records, archive.records[-1].evidence)
        candidate, selection_reason = _select_candidate(remaining, route)
        remaining.remove(candidate)
        trace.emit(
            "candidate_selected",
            candidate_id=candidate_id,
            patch_file=str(candidate.path),
            candidate_label=candidate.label,
            candidate_intent=candidate.intent,
            compatible_routes=list(candidate.compatible_routes),
            route=route,
            selection_reason=selection_reason,
        )
        result = run_task(task_dir, run_dir / "candidates" / candidate_id, patch_file=candidate.path)
        evidence = build_evidence_packet(result.gates)
        failure_type = classify_failure(result.gates)
        fingerprint = _failure_fingerprint(result.gates)
        record = CandidateRecord(
            candidate_id=candidate_id,
            patch_file=str(candidate.path),
            status=result.status,
            route=route,
            failure_type=failure_type,
            score=_score_result(result.status, failure_type),
            gates=tuple({"name": gate.name, "passed": gate.passed} for gate in result.gates),
            parent_id=archive.best_parent_id(),
            generation=archive.next_generation(),
            touched_files=result.touched_files,
            candidate_label=candidate.label,
            candidate_intent=candidate.intent,
            compatible_routes=candidate.compatible_routes,
            selection_reason=selection_reason,
            rationale=_rationale(route, evidence),
            verifier_results=tuple(_verifier_result(gate) for gate in result.gates),
            failure_fingerprint=fingerprint,
            token_estimate=_estimate_tokens(route, evidence),
            duration_seconds=result.duration_seconds,
            evidence=evidence,
        )
        archive.add(record)
        trace.emit(
            "candidate_evaluated",
            candidate_id=candidate_id,
            status=record.status,
            failure_type=failure_type,
            score=record.score,
            evidence_summary=evidence.summary if evidence else None,
        )

        if result.passed:
            stop_reason = "promoted"
            break
        if archive.repeated_failures(fingerprint or failure_type) >= budget.max_repeated_failures:
            stop_reason = "repeated_failure_limit"
            trace.emit("search_stopped", reason=stop_reason, failure_type=failure_type, fingerprint=fingerprint)
            break
    else:
        if not remaining:
            stop_reason = "candidate_pool_exhausted"

    total_tokens = sum(record.token_estimate for record in archive.records)
    wall_seconds = time.monotonic() - started
    archive_path = run_dir / "candidate_archive.json"
    archive.save(
        archive_path,
        run=build_run_metadata(
            task,
            archive.records,
            status="promoted" if archive.records and archive.records[-1].status == "promoted" else "stopped",
            stop_reason=stop_reason,
            total_tokens=total_tokens,
            wall_seconds=wall_seconds,
            budget=budget,
            model=model,
            provider=provider,
            temperature=temperature,
        ),
    )
    status = "promoted" if archive.records and archive.records[-1].status == "promoted" else "stopped"
    trace.emit(
        "search_finished",
        task_id=task.task_id,
        status=status,
        stop_reason=stop_reason,
        archive_path=str(archive_path),
        evaluated_candidates=len(archive.records),
    )
    return SearchResult(
        task_id=task.task_id,
        status=status,
        stop_reason=stop_reason,
        trace_path=trace_path,
        archive_path=archive_path,
        records=archive.records,
        total_tokens=total_tokens,
        wall_seconds=wall_seconds,
    )


def _score_result(status: str, failure_type: str | None) -> float:
    if status == "promoted":
        return 1.0
    if failure_type == "release_gate_regressions":
        return 0.6
    if failure_type == "visible_tests":
        return 0.4
    if failure_type in {"python_ast", "secret_scan", "scope_guard"}:
        return 0.1
    return 0.0


def _candidate_spec(candidate: Path | CandidateSpec) -> CandidateSpec:
    if isinstance(candidate, CandidateSpec):
        return candidate
    return CandidateSpec(
        path=candidate,
        label=candidate.stem,
        compatible_routes=(),
        generator="path_only",
        intent="ordered_candidate",
    )


def _select_candidate(remaining: list[CandidateSpec], route: str) -> tuple[CandidateSpec, str]:
    if route == "initial_repair":
        return remaining[0], "initial_order"
    ranked = sorted(enumerate(remaining), key=lambda item: _candidate_rank(item[0], item[1], route))
    _, selected = ranked[0]
    if route in selected.compatible_routes:
        return selected, f"metadata_route_match:{route}"
    return selected, "ordered_fallback_no_route_metadata"


def _candidate_rank(index: int, candidate: CandidateSpec, route: str) -> tuple[int, int]:
    if route in candidate.compatible_routes:
        return (0, index)
    if "general_repair" in candidate.compatible_routes:
        return (10, index)
    return (20, index)


def _failure_fingerprint(gates) -> str | None:
    for gate in gates:
        if not gate.passed:
            return gate.fingerprint or gate.name
    return None


def _rationale(route: str, evidence) -> str:
    if evidence is None:
        return "candidate passed all hard gates"
    return f"{route}: respond to {evidence.summary}"


def _verifier_result(gate) -> dict[str, object]:
    return {
        "name": gate.name,
        "passed": gate.passed,
        "fingerprint": gate.fingerprint,
    }


def _estimate_tokens(route: str, evidence) -> int:
    text = route
    if evidence is not None:
        text += f" {evidence.summary} {evidence.details}"
    return max(1, len(text) // 4)


def _wall_budget_exhausted(started: float, budget: SearchBudget) -> bool:
    if budget.max_wall_seconds is None:
        return False
    return (time.monotonic() - started) >= budget.max_wall_seconds
