from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .benchmark import run_benchmark_suite
from .controller import run_search
from .llm import (
    OPENROUTER_MODEL,
    VLLM_DEFAULT_BASE_URL,
    VLLM_DEFAULT_MODEL,
    run_llm_search,
)
from .models import SearchBudget
from .policy import run_policy_experiment
from .report import generate_html_report


def run_demo(
    run_dir: Path,
    use_llm: bool = False,
    llm_provider: str = "openrouter",
    llm_model: str | None = None,
    llm_base_url: str | None = None,
    llm_api_key: str | None = None,
    llm_cache_dir: str | Path | None = None,
    llm_temperature: float | None = None,
    llm_seed: int | None = None,
    llm_reasoning: bool = True,
    llm_max_candidates: int = 6,
    llm_max_repeated_failures: int = 6,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    task_dir = repo_root / "tasks" / "smoke_markdown_parser"
    benchmark_suite = repo_root / "benchmarks" / "repair_routing_suite.json"
    policy_config = repo_root / "benchmarks" / "policy_experiment_suite.json"
    run_dir.mkdir(parents=True, exist_ok=True)

    if use_llm:
        resolved_model = llm_model or (
            OPENROUTER_MODEL if llm_provider == "openrouter" else VLLM_DEFAULT_MODEL
        )
        resolved_base_url = None
        if llm_provider in {"vllm", "openai"}:
            resolved_base_url = llm_base_url or os.environ.get("VLLM_BASE_URL") or VLLM_DEFAULT_BASE_URL
        search = run_llm_search(
            task_dir,
            run_dir / "hero",
            budget=SearchBudget(
                max_candidates=llm_max_candidates,
                max_repeated_failures=llm_max_repeated_failures,
            ),
            model=llm_model,
            reasoning_enabled=llm_reasoning,
            provider=llm_provider,
            base_url=llm_base_url,
            api_key=llm_api_key,
            cache_dir=llm_cache_dir,
            temperature=llm_temperature,
            seed=llm_seed,
        )
    else:
        resolved_model = None
        resolved_base_url = None
        search = run_search(task_dir, run_dir / "hero")
    hero_report = generate_html_report(run_dir / "hero", run_dir / "hero_report.html")

    benchmark = run_benchmark_suite(benchmark_suite, run_dir / "benchmark")
    benchmark_report = generate_html_report(run_dir / "benchmark", run_dir / "benchmark_report.html")

    policy = run_policy_experiment(policy_config, run_dir / "policy")

    summary = {
        "llm_required": use_llm,
        "llm_mode": llm_provider if use_llm else "offline_curated_patches",
        "llm_model": resolved_model,
        "llm_base_url": resolved_base_url,
        "llm_reasoning": llm_reasoning if use_llm else None,
        "why_no_llm": (
            "This MVP uses deterministic candidate patches so the repair-search controller, "
            "verification gates, archive, benchmark, reports, and policy experiment are reproducible offline."
        )
        if not use_llm
        else None,
        "where_llm_fits": (
            "In --llm mode, the configured "
            f"{'OpenRouter API' if llm_provider == 'openrouter' else 'OpenAI-compatible server'}"
            f"{' at ' + resolved_base_url if resolved_base_url else ''} generates hero candidate patches before GateGRPO gates/search/report evaluate them."
        ),
        "hero": {
            "status": search.status,
            "stop_reason": search.stop_reason,
            "attempts": len(search.records),
            "first_failure": search.records[0].failure_type if search.records else None,
            "final_route": search.records[-1].route if search.records else None,
            "report": str(hero_report),
            "archive": str(search.archive_path),
        },
        "benchmark": {
            "summary": benchmark["summary"],
            "report": str(benchmark_report),
            "results": str(run_dir / "benchmark" / "benchmark_results.json"),
        },
        "policy": {
            "selected_policy": policy["selected_policy"]["id"],
            "summary": str(run_dir / "policy" / "policy_experiment_summary.md"),
            "results": str(run_dir / "policy" / "policy_experiment_results.json"),
        },
    }
    (run_dir / "demo_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary
