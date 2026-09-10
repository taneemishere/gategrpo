from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from .controller import run_search
from .critic import CleanContextCritic
from .evidence import build_evidence_packet, classify_failure
from .llm import build_llm_client, run_llm_search
from .models import CandidateSpec, SearchBudget
from .pricing import load_price_schedule
from .report import generate_model_matrix_report
from .runner import run_task
from .task import load_task


BASELINES = (
    "single_shot",
    "linear_retry",
    "clean_context_review",
    "evidence_aware_review",
    "archive_no_routing",
    "full_gategrpo",
)


def run_benchmark_suite(
    suite_path: Path,
    run_dir: Path,
    baselines: tuple[str, ...] = BASELINES,
    repetitions: int = 1,
    budget: SearchBudget | None = None,
    candidate_generator: Callable[[Path, Path], list[CandidateSpec]] | None = None,
) -> dict[str, Any]:
    suite = _load_suite(suite_path)
    budget = budget or SearchBudget()
    run_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    generated_pools: dict[Path, list[CandidateSpec]] = {}
    candidate_generator_name: str | None = "llm_live" if candidate_generator is not None else None

    for case in suite["cases"]:
        case_repetitions = int(case.get("repetitions", repetitions))
        task_dir = _resolve_task_dir(suite_path, case["task_dir"])
        if candidate_generator is None:
            candidates = [_candidate_spec(task_dir, patch) for patch in case["candidate_patches"]]
        else:
            if task_dir not in generated_pools:
                task = load_task(task_dir)
                generated_pools[task_dir] = candidate_generator(task_dir, run_dir / task.task_id)
            candidates = generated_pools[task_dir]
        for repetition in range(1, case_repetitions + 1):
            for baseline in baselines:
                if baseline not in BASELINES:
                    raise ValueError(f"unknown baseline: {baseline}")
                result = _run_baseline(
                    baseline=baseline,
                    task_dir=task_dir,
                    candidate_files=candidates,
                    run_dir=run_dir / case["id"] / baseline / f"rep_{repetition:03d}",
                    budget=budget,
                )
                result.update(
                    {
                        "suite": suite["name"],
                        "case_id": case["id"],
                        "repetition": repetition,
                        "expected": case.get("expected", {}),
                        "candidate_generator": candidate_generator_name,
                    }
                )
                results.append(result)

    summary = _summarize(results, candidate_generator=candidate_generator_name)
    payload = {"suite": suite["name"], "results": results, "summary": summary}
    if candidate_generator_name is not None:
        payload["candidate_generator"] = candidate_generator_name
    (run_dir / "benchmark_results.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    (run_dir / "benchmark_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (run_dir / "benchmark_summary.md").write_text(_summary_markdown(suite["name"], summary), encoding="utf-8")
    return payload


def run_llm_benchmark_suite(
    suite_path: Path,
    run_dir: Path,
    *,
    provider: str = "vllm",
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    reasoning_enabled: bool = False,
    temperature: float | None = None,
    seed: int | None = None,
    budget: SearchBudget | None = None,
    repetitions: int = 3,
    cache_dir: str | Path | None = None,
    price_schedule: dict[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    suite = _load_suite(suite_path)
    budget = budget or SearchBudget(max_candidates=6, max_repeated_failures=6)
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved_price_schedule = (
        price_schedule if isinstance(price_schedule, dict) else load_price_schedule(price_schedule)
    )
    client = build_llm_client(
        provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
        reasoning_enabled=reasoning_enabled,
        temperature=temperature,
        seed=seed,
        cache_dir=cache_dir,
    )

    families: list[tuple[str, Path]] = []
    seen_task_dirs: set[Path] = set()
    for case in suite["cases"]:
        task_dir = _resolve_task_dir(suite_path, case["task_dir"])
        if task_dir in seen_task_dirs:
            continue
        seen_task_dirs.add(task_dir)
        task = load_task(task_dir)
        families.append((task.task_id, task_dir))

    results: list[dict[str, Any]] = []
    for task_id, task_dir in families:
        for repetition in range(1, repetitions + 1):
            result = run_llm_search(
                task_dir,
                run_dir / task_id / f"rep_{repetition:03d}",
                budget=budget,
                client=client,
                provider=provider,
                temperature=temperature,
                seed=seed,
                price_schedule=resolved_price_schedule,
            )
            failure_types = [record.failure_type for record in result.records if record.failure_type]
            results.append(
                {
                    "task_id": task_id,
                    "repetition": repetition,
                    "status": result.status,
                    "stop_reason": result.stop_reason,
                    "attempts": len(result.records),
                    "promoted": result.status == "promoted",
                    "failure_types": failure_types,
                    "total_tokens": result.total_tokens,
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "token_source": result.token_source,
                    "cost": result.cost,
                    "currency": result.currency,
                    "wall_seconds": result.wall_seconds,
                }
            )

    summary: dict[str, Any] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        grouped[row["task_id"]].append(row)
    for task_id, rows in grouped.items():
        summary[task_id] = _llm_task_summary(rows)
    summary["all_tasks"] = _llm_task_summary(results)

    payload = {"suite": suite["name"], "model": client.model, "provider": provider, "results": results, "summary": summary}
    (run_dir / "llm_benchmark_results.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    (run_dir / "llm_benchmark_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (run_dir / "llm_benchmark_summary.md").write_text(_llm_summary_markdown(suite["name"], summary), encoding="utf-8")
    return payload


def run_llm_model_matrix(
    suite_path: Path,
    run_dir: Path,
    model_specs: list[dict[str, Any]],
    *,
    budget: SearchBudget | None = None,
    repetitions: int = 3,
    cache_dir: str | Path | None = None,
    price_schedule: dict[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    """Run the live LLM benchmark across several model families as one experiment.

    ``model_specs`` is a list of dicts, each describing one column of the matrix::

        {"id": "qwen3-coder-30b", "provider": "vllm", "model": "Qwen3-Coder-30B-A3B-Instruct",
         "base_url": "...", "api_key": None, "reasoning_enabled": False, "temperature": None}

    Each model runs the same suite on the same hard gates; results are merged into
    a single comparison table so different model families are compared as one
    experiment rather than manual repeated invocations.
    """
    if not model_specs:
        raise ValueError("run_llm_model_matrix requires at least one model spec")
    suite = _load_suite(suite_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved_price_schedule = (
        price_schedule if isinstance(price_schedule, dict) else load_price_schedule(price_schedule)
    )

    matrix: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for spec in model_specs:
        model_id = _model_matrix_id(spec, seen_ids)
        seen_ids.add(model_id)
        provider = str(spec.get("provider", "vllm"))
        model_run_dir = run_dir / model_id
        payload = run_llm_benchmark_suite(
            suite_path=suite_path,
            run_dir=model_run_dir,
            provider=provider,
            base_url=spec.get("base_url"),
            model=spec.get("model"),
            api_key=spec.get("api_key"),
            reasoning_enabled=bool(spec.get("reasoning_enabled", False)),
            temperature=spec.get("temperature"),
            seed=spec.get("seed"),
            budget=budget,
            repetitions=repetitions,
            cache_dir=cache_dir,
            price_schedule=resolved_price_schedule,
        )
        matrix.append(
            {
                "id": model_id,
                "provider": provider,
                "model": payload["model"],
                "run_dir": str(model_run_dir),
                "overall": payload["summary"].get("all_tasks", {}),
                "per_task": {
                    task_id: row for task_id, row in payload["summary"].items() if task_id != "all_tasks"
                },
            }
        )

    result = {"suite": suite["name"], "matrix": matrix}
    (run_dir / "model_matrix_results.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    (run_dir / "model_matrix_summary.md").write_text(_model_matrix_markdown(suite["name"], matrix), encoding="utf-8")
    generate_model_matrix_report(run_dir, run_dir / "model_matrix_report.html")
    return result


def _model_matrix_id(spec: dict[str, Any], seen_ids: set[str]) -> str:
    base = str(spec.get("id") or spec.get("model") or spec.get("provider") or "model")
    candidate = "".join(char if char.isalnum() or char in "-_." else "_" for char in base)
    model_id = candidate
    suffix = 2
    while model_id in seen_ids:
        model_id = f"{candidate}_{suffix}"
        suffix += 1
    return model_id


def _model_matrix_markdown(suite_name: str, matrix: list[dict[str, Any]]) -> str:
    lines = [
        f"# GateGRPO live LLM model matrix: {suite_name}",
        "",
        "| model | provider | runs | solve@budget | 95% CI | avg attempts | avg tokens | token source | avg cost | avg wall (s) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for entry in matrix:
        overall = entry.get("overall", {})
        ci = overall.get("solve_at_budget_ci95", (0.0, 0.0))
        lines.append(
            "| "
            + " | ".join(
                [
                    entry["model"],
                    entry["provider"],
                    str(overall.get("runs", 0)),
                    str(overall.get("solve_at_budget", 0.0)),
                    f"{ci[0]}-{ci[1]}",
                    str(overall.get("avg_attempts", 0.0)),
                    str(overall.get("avg_tokens", 0.0)),
                    str(overall.get("token_source", "estimate")),
                    _format_avg_cost(overall),
                    str(overall.get("avg_wall_seconds", 0.0)),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _run_baseline(
    baseline: str,
    task_dir: Path,
    candidate_files: list[CandidateSpec],
    run_dir: Path,
    budget: SearchBudget,
) -> dict[str, Any]:
    if baseline == "full_gategrpo":
        result = run_search(task_dir, run_dir, budget, candidate_files)
        failure_types = [record.failure_type for record in result.records if record.failure_type]
        return {
            "baseline": baseline,
            "status": result.status,
            "stop_reason": result.stop_reason,
            "attempts": len(result.records),
            "failure_types": failure_types,
            "evidence_packets": sum(1 for record in result.records if record.evidence is not None),
            "selection_reasons": [record.selection_reason for record in result.records],
            "trace_path": str(result.trace_path),
            "archive_path": str(result.archive_path),
        }

    if baseline == "single_shot":
        selected = candidate_files[:1]
        repeated_limit = 0
    elif baseline == "linear_retry":
        selected = candidate_files[: budget.max_candidates]
        repeated_limit = 0
    elif baseline == "clean_context_review":
        selected = candidate_files[: budget.max_candidates]
        repeated_limit = 0
    elif baseline == "evidence_aware_review":
        return _run_evidence_aware_review(task_dir, candidate_files, run_dir, budget)
    else:
        selected = candidate_files[: budget.max_candidates]
        repeated_limit = 0

    failure_counts: Counter[str] = Counter()
    failure_types: list[str] = []
    status = "stopped"
    stop_reason = "candidate_pool_exhausted"
    attempts = 0
    for index, patch_file in enumerate(selected, start=1):
        attempts += 1
        result = run_task(task_dir, run_dir / f"candidate_{index:03d}", patch_file=patch_file.path)
        failure_type = classify_failure(result.gates)
        if result.passed:
            status = "promoted"
            stop_reason = "promoted"
            break
        if failure_type:
            failure_types.append(failure_type)
            failure_counts[failure_type] += 1
        if repeated_limit and failure_type and failure_counts[failure_type] >= repeated_limit:
            stop_reason = "repeated_failure_limit"
            break

    return {
        "baseline": baseline,
        "status": status,
        "stop_reason": stop_reason,
        "attempts": attempts,
        "failure_types": failure_types,
        "evidence_packets": 0,
        "selection_reasons": ["ordered_baseline" for _ in range(attempts)],
        "trace_path": str(run_dir),
        "archive_path": None,
    }


def _run_evidence_aware_review(
    task_dir: Path,
    candidate_files: list[CandidateSpec],
    run_dir: Path,
    budget: SearchBudget,
) -> dict[str, Any]:
    critic = CleanContextCritic()
    remaining = list(candidate_files)
    failure_types: list[str] = []
    selection_reasons: list[str] = []
    evidence_packets = 0
    status = "stopped"
    stop_reason = "candidate_budget_exhausted"
    attempts = 0
    evidence = None

    while remaining and attempts < budget.max_candidates:
        attempts += 1
        route = "initial_repair" if attempts == 1 else critic.route_next([], evidence)
        candidate, selection_reason = _select_evidence_candidate(remaining, route)
        remaining.remove(candidate)
        selection_reasons.append(selection_reason)

        result = run_task(task_dir, run_dir / f"candidate_{attempts:03d}", patch_file=candidate.path)
        failure_type = classify_failure(result.gates)
        if result.passed:
            status = "promoted"
            stop_reason = "promoted"
            break
        if failure_type:
            failure_types.append(failure_type)
        evidence = build_evidence_packet(result.gates)
        if evidence is not None:
            evidence_packets += 1
    else:
        if not remaining:
            stop_reason = "candidate_pool_exhausted"

    return {
        "baseline": "evidence_aware_review",
        "status": status,
        "stop_reason": stop_reason,
        "attempts": attempts,
        "failure_types": failure_types,
        "evidence_packets": evidence_packets,
        "selection_reasons": selection_reasons,
        "trace_path": str(run_dir),
        "archive_path": None,
    }


def _select_evidence_candidate(remaining: list[CandidateSpec], route: str) -> tuple[CandidateSpec, str]:
    if route == "initial_repair":
        return remaining[0], "initial_order"
    ranked = sorted(enumerate(remaining), key=lambda item: _evidence_candidate_rank(item[0], item[1], route))
    _, selected = ranked[0]
    if route in selected.compatible_routes:
        return selected, f"metadata_route_match:{route}"
    return selected, "ordered_fallback_no_route_metadata"


def _evidence_candidate_rank(index: int, candidate: CandidateSpec, route: str) -> tuple[int, int]:
    if route in candidate.compatible_routes:
        return (0, index)
    if "general_repair" in candidate.compatible_routes:
        return (10, index)
    return (20, index)


def _summarize(results: list[dict[str, Any]], candidate_generator: str | None = None) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[result["baseline"]].append(result)

    summary: dict[str, Any] = {}
    for baseline, rows in sorted(grouped.items()):
        solved = sum(1 for row in rows if row["status"] == "promoted")
        attempts = sum(int(row["attempts"]) for row in rows)
        unsolved = [row for row in rows if row["status"] != "promoted"]
        failure_types = [failure for row in rows for failure in row["failure_types"]]
        repeated_failures = sum(_repeated_failure_count(row["failure_types"]) for row in rows)
        summary[baseline] = {
            "runs": len(rows),
            "solved": solved,
            "solve_at_budget": round(solved / len(rows), 4) if rows else 0.0,
            "solve_at_budget_ci95": _wilson_ci(solved, len(rows)),
            "attempts": attempts,
            "avg_attempts": round(attempts / len(rows), 4) if rows else 0.0,
            "unsolved_task_cost": sum(int(row["attempts"]) for row in unsolved),
            "regressions_blocked": failure_types.count("release_gate_regressions"),
            "invalid_patches_rejected": sum(
                failure_types.count(name) for name in ("patch_applies", "scope_guard", "python_ast", "secret_scan")
            ),
            "repeated_failure_rate": round(repeated_failures / attempts, 4) if attempts else 0.0,
            "evidence_packets_admitted": sum(int(row["evidence_packets"]) for row in rows),
            "metadata_route_matches": sum(
                reason.startswith("metadata_route_match:")
                for row in rows
                for reason in row.get("selection_reasons", [])
            ),
        }
        if candidate_generator is not None:
            summary[baseline]["candidate_generator"] = candidate_generator
    return summary


def _repeated_failure_count(failure_types: list[str]) -> int:
    seen: set[str] = set()
    repeated = 0
    for failure_type in failure_types:
        if failure_type in seen:
            repeated += 1
        seen.add(failure_type)
    return repeated


def _summary_markdown(suite_name: str, summary: dict[str, Any]) -> str:
    lines = [
        f"# GateGRPO benchmark: {suite_name}",
        "",
        "| baseline | solve@budget | 95% CI | avg attempts | regressions blocked | invalid rejected | evidence packets | route metadata matches |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for baseline, row in summary.items():
        lines.append(
            "| "
            + " | ".join(
                [
                    baseline,
                    str(row["solve_at_budget"]),
                    f"{row['solve_at_budget_ci95'][0]}-{row['solve_at_budget_ci95'][1]}",
                    str(row["avg_attempts"]),
                    str(row["regressions_blocked"]),
                    str(row["invalid_patches_rejected"]),
                    str(row["evidence_packets_admitted"]),
                    str(row["metadata_route_matches"]),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _llm_task_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    solved = sum(1 for row in rows if row["promoted"])
    attempts = sum(int(row["attempts"]) for row in rows)
    runs = len(rows)
    token_values = [int(row.get("total_tokens", 0)) for row in rows]
    wall_values = [float(row.get("wall_seconds", 0.0)) for row in rows]
    total_tokens = sum(token_values)
    total_prompt_tokens = sum(int(row.get("prompt_tokens", 0)) for row in rows)
    total_completion_tokens = sum(int(row.get("completion_tokens", 0)) for row in rows)
    total_wall_seconds = sum(wall_values)
    costs = [row["cost"] for row in rows if row.get("cost") is not None]
    total_cost = round(sum(float(cost) for cost in costs), 6) if costs else None
    summary = {
        "runs": runs,
        "solved": solved,
        "solve_at_budget": round(solved / runs, 4) if runs else 0.0,
        "solve_at_budget_ci95": _wilson_ci(solved, runs),
        "avg_attempts": round(attempts / runs, 4) if runs else 0.0,
        "total_tokens": total_tokens,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "avg_tokens": round(total_tokens / runs, 1) if runs else 0.0,
        "tokens_stdev": _stdev(token_values),
        "token_source": _dominant_token_source(rows),
        "total_wall_seconds": round(total_wall_seconds, 4),
        "avg_wall_seconds": round(total_wall_seconds / runs, 2) if runs else 0.0,
        "wall_seconds_stdev": _stdev(wall_values),
    }
    if total_cost is not None:
        currency = _dominant_currency(rows)
        summary["total_cost"] = total_cost
        summary["avg_cost"] = round(total_cost / runs, 6) if runs else 0.0
        summary["currency"] = currency
    return summary


def _stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return round(statistics.stdev(values), 4)


def _dominant_currency(rows: list[dict[str, Any]]) -> str:
    currencies = [
        str(row["currency"])
        for row in rows
        if row.get("cost") is not None and row.get("currency")
    ]
    return currencies[0] if currencies else "USD"


def _format_avg_cost(summary: dict[str, Any]) -> str:
    avg_cost = summary.get("avg_cost")
    if avg_cost is None:
        return "n/a"
    return f"{avg_cost} {summary.get('currency', 'USD')}"


def _dominant_token_source(rows: list[dict[str, Any]]) -> str:
    sources = [str(row.get("token_source", "estimate")) for row in rows]
    if sources and all(source == "provider_usage" for source in sources):
        return "provider_usage"
    if sources and any(source == "provider_usage" for source in sources):
        return "mixed"
    return "estimate"


def _llm_summary_markdown(suite_name: str, summary: dict[str, Any]) -> str:
    lines = [
        f"# GateGRPO live LLM benchmark: {suite_name}",
        "",
        "| task | runs | solve@budget | 95% CI | avg attempts | avg tokens | token source | avg cost | avg wall (s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for task, row in summary.items():
        lines.append(
            "| "
            + " | ".join(
                [
                    task,
                    str(row["runs"]),
                    str(row["solve_at_budget"]),
                    f"{row['solve_at_budget_ci95'][0]}-{row['solve_at_budget_ci95'][1]}",
                    str(row["avg_attempts"]),
                    str(row["avg_tokens"]),
                    str(row.get("token_source", "estimate")),
                    _format_avg_cost(row),
                    str(row["avg_wall_seconds"]),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _load_suite(suite_path: Path) -> dict[str, Any]:
    with suite_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_task_dir(suite_path: Path, task_dir: str) -> Path:
    path = Path(task_dir)
    if path.is_absolute():
        return path
    repo_root = suite_path.resolve().parent.parent
    return repo_root / path


def _wilson_ci(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    z = 1.96
    p_hat = successes / total
    denom = 1 + z * z / total
    center = (p_hat + z * z / (2 * total)) / denom
    margin = z * ((p_hat * (1 - p_hat) + z * z / (4 * total)) / total) ** 0.5 / denom
    return (round(max(0.0, center - margin), 4), round(min(1.0, center + margin), 4))


def _candidate_spec(task_dir: Path, candidate: str | dict[str, Any]) -> CandidateSpec:
    if isinstance(candidate, str):
        return CandidateSpec(
            path=_resolve_patch(task_dir, candidate),
            label=Path(candidate).stem,
            compatible_routes=(),
            generator="path_only",
            intent="ordered_candidate",
        )
    return CandidateSpec(
        path=_resolve_patch(task_dir, str(candidate["patch"])),
        label=str(candidate.get("label", Path(str(candidate["patch"])).stem)),
        compatible_routes=tuple(str(route) for route in candidate.get("compatible_routes", [])),
        generator=str(candidate.get("generator", "synthetic_generator")),
        intent=str(candidate.get("intent", "unspecified")),
    )


def _resolve_patch(task_dir: Path, patch: str) -> Path:
    path = Path(patch)
    if path.is_absolute():
        return path
    return task_dir / path
