from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from gategrpo.benchmark import (
    _llm_task_summary,
    run_llm_benchmark_suite,
    run_llm_model_matrix,
)
from gategrpo.generators import generate_candidate_pool
from gategrpo.models import CandidateSpec, SearchBudget, SearchResult
from gategrpo.task import load_task


ROOT = Path(__file__).resolve().parents[1]
SMOKE_SUITE = ROOT / "benchmarks" / "repair_routing_suite.json"


def test_run_llm_benchmark_suite_deduplicates_task_families_and_reuses_client(tmp_path, monkeypatch):
    client = SimpleNamespace(model="fake-llm")
    build_calls: list[tuple[object, ...]] = []
    run_calls: list[tuple[Path, Path, int, object]] = []

    def fake_build_llm_client(provider, *, model, base_url, api_key, reasoning_enabled, temperature=None, seed=None, cache_dir=None):
        build_calls.append((provider, model, base_url, api_key, reasoning_enabled, temperature, cache_dir))
        return client

    def fake_run_llm_search(task_dir, run_dir, budget, client=None, **kwargs):
        run_calls.append((task_dir, run_dir, budget.max_candidates, client))
        task = load_task(task_dir)
        repetition = int(run_dir.name.split("_")[-1])
        return SearchResult(
            task_id=task.task_id,
            status="promoted",
            stop_reason="promoted",
            trace_path=run_dir / "trace.jsonl",
            archive_path=run_dir / "archive.json",
            records=[SimpleNamespace(failure_type=None)],
            total_tokens=100 * repetition,
            wall_seconds=0.1 * repetition,
        )

    monkeypatch.setattr("gategrpo.benchmark.build_llm_client", fake_build_llm_client)
    monkeypatch.setattr("gategrpo.benchmark.run_llm_search", fake_run_llm_search)

    payload = run_llm_benchmark_suite(
        SMOKE_SUITE,
        tmp_path / "llm_benchmark",
        provider="vllm",
        base_url="http://localhost:8000/v1",
        model=None,
        api_key=None,
        reasoning_enabled=False,
        budget=SearchBudget(max_candidates=6, max_repeated_failures=6),
        repetitions=2,
    )

    assert build_calls == [("vllm", None, "http://localhost:8000/v1", None, False, None, None)]
    assert len(run_calls) == 8
    assert len({task_dir for task_dir, _, _, _ in run_calls}) == 4
    assert all(client_arg is client for _, _, _, client_arg in run_calls)
    assert payload["model"] == "fake-llm"
    assert payload["provider"] == "vllm"
    assert payload["summary"]["all_tasks"]["runs"] == 8
    assert payload["summary"]["all_tasks"]["solved"] == 8
    assert payload["summary"]["all_tasks"]["solve_at_budget"] == 1.0
    assert payload["summary"]["all_tasks"]["avg_attempts"] == 1.0
    assert payload["summary"]["all_tasks"]["total_tokens"] == 1200
    assert payload["summary"]["all_tasks"]["avg_tokens"] == 150.0
    assert payload["summary"]["all_tasks"]["total_wall_seconds"] == 1.2
    assert payload["summary"]["all_tasks"]["avg_wall_seconds"] == 0.15
    assert set(payload["summary"]) == {
        "config_loader",
        "range_parser",
        "smoke_markdown_parser",
        "string_slugifier",
        "all_tasks",
    }
    for task_id in ("config_loader", "range_parser", "smoke_markdown_parser", "string_slugifier"):
        assert payload["summary"][task_id]["runs"] == 2
        assert payload["summary"][task_id]["solved"] == 2
        assert payload["summary"][task_id]["solve_at_budget"] == 1.0
        assert payload["summary"][task_id]["avg_attempts"] == 1.0
        assert payload["summary"][task_id]["total_tokens"] == 300
        assert payload["summary"][task_id]["avg_tokens"] == 150.0
        assert payload["summary"][task_id]["total_wall_seconds"] == 0.3
        assert payload["summary"][task_id]["avg_wall_seconds"] == 0.15

    assert len(payload["results"]) == 8
    assert all(row["promoted"] for row in payload["results"])
    assert payload["results"][0]["total_tokens"] == 100
    assert payload["results"][0]["wall_seconds"] == 0.1
    assert (tmp_path / "llm_benchmark" / "llm_benchmark_results.json").exists()
    assert (tmp_path / "llm_benchmark" / "llm_benchmark_summary.json").exists()
    assert (tmp_path / "llm_benchmark" / "llm_benchmark_summary.md").exists()


def test_run_llm_benchmark_suite_summary_handles_non_promotions(tmp_path, monkeypatch):
    client = SimpleNamespace(model="fake-llm")

    def fake_build_llm_client(*args, **kwargs):
        return client

    def fake_run_llm_search(task_dir, run_dir, budget, client=None, **kwargs):
        task = load_task(task_dir)
        repetition = int(run_dir.name.split("_")[-1])
        return SearchResult(
            task_id=task.task_id,
            status="stopped",
            stop_reason="budget_exhausted",
            trace_path=run_dir / "trace.jsonl",
            archive_path=run_dir / "archive.json",
            records=[SimpleNamespace(failure_type="visible_tests"), SimpleNamespace(failure_type="visible_tests")],
            total_tokens=50 * repetition,
            wall_seconds=0.25 * repetition,
        )

    monkeypatch.setattr("gategrpo.benchmark.build_llm_client", fake_build_llm_client)
    monkeypatch.setattr("gategrpo.benchmark.run_llm_search", fake_run_llm_search)

    payload = run_llm_benchmark_suite(
        SMOKE_SUITE,
        tmp_path / "llm_benchmark",
        repetitions=1,
    )

    assert payload["summary"]["all_tasks"]["solve_at_budget"] == 0.0
    assert 0.0 <= payload["summary"]["all_tasks"]["solve_at_budget"] <= 1.0
    assert 0.0 <= payload["summary"]["config_loader"]["solve_at_budget"] <= 1.0
    assert payload["summary"]["all_tasks"]["total_tokens"] == 50 * 4
    assert payload["summary"]["all_tasks"]["avg_tokens"] == 50.0
    assert payload["summary"]["all_tasks"]["total_wall_seconds"] > 0
    assert payload["results"][0]["total_tokens"] == 50
    assert payload["results"][0]["wall_seconds"] == 0.25


def test_multifile_suite_includes_new_tasks_and_two_allowed_paths():
    suite = json.loads((ROOT / "benchmarks" / "multifile_suite.json").read_text(encoding="utf-8"))
    task_dirs = {case["task_dir"] for case in suite["cases"]}
    assert {"tasks/csv_report", "tasks/pricing_cart", "tasks/order_service"} <= task_dirs
    assert len(load_task(ROOT / "tasks" / "csv_report").allowed_paths) == 2
    assert len(load_task(ROOT / "tasks" / "pricing_cart").allowed_paths) == 2
    assert len(load_task(ROOT / "tasks" / "order_service").allowed_paths) == 3


def test_generate_candidate_pool_builds_real_metadata_and_skips_empty_records(tmp_path, monkeypatch):
    task_dir = ROOT / "tasks" / "csv_report"
    generation_root = tmp_path / "candidate_generation"
    valid_patch = tmp_path / "valid.patch"
    promoted_patch = tmp_path / "promoted.patch"
    empty_patch = tmp_path / "empty.patch"
    missing_patch = tmp_path / "missing.patch"
    valid_patch.write_text("diff --git a/rows.py b/rows.py\n", encoding="utf-8")
    promoted_patch.write_text("diff --git a/summary.py b/summary.py\n", encoding="utf-8")
    empty_patch.write_text("", encoding="utf-8")

    def fake_run_llm_search(*args, **kwargs):
        return SearchResult(
            task_id="csv_report",
            status="stopped",
            stop_reason="budget_exhausted",
            trace_path=generation_root / "trace.jsonl",
            archive_path=generation_root / "archive.json",
            records=[
                SimpleNamespace(
                    candidate_id="candidate_001",
                    patch_file=str(valid_patch),
                    route="behavior_repair",
                    failure_type="visible_tests",
                    status="stopped",
                    metadata_source="controller",
                    candidate_intent="",
                    compatible_routes=(),
                ),
                SimpleNamespace(
                    candidate_id="candidate_002",
                    patch_file=str(empty_patch),
                    route="regression_repair",
                    failure_type="release_gate_regressions",
                    status="stopped",
                    metadata_source="controller",
                    candidate_intent="",
                    compatible_routes=(),
                ),
                SimpleNamespace(
                    candidate_id="candidate_003",
                    patch_file=str(missing_patch),
                    route="scope_repair",
                    failure_type=None,
                    status="promoted",
                    metadata_source="controller",
                    candidate_intent="",
                    compatible_routes=(),
                ),
                SimpleNamespace(
                    candidate_id="candidate_004",
                    patch_file=str(promoted_patch),
                    route="general_repair",
                    failure_type=None,
                    status="promoted",
                    metadata_source="controller",
                    candidate_intent="",
                    compatible_routes=(),
                ),
            ],
        )

    monkeypatch.setattr("gategrpo.generators.run_llm_search", fake_run_llm_search)

    pool = generate_candidate_pool(
        task_dir,
        generation_root,
        client=SimpleNamespace(model="fake-llm", reasoning_enabled=False),
        budget=SearchBudget(max_candidates=6, max_repeated_failures=6),
    )

    assert [spec.label for spec in pool] == ["candidate_001", "candidate_004"]
    assert [spec.generator for spec in pool] == ["llm_live", "llm_live"]
    assert [spec.intent for spec in pool] == ["visible_tests", "promoted"]
    assert [spec.compatible_routes for spec in pool] == [("behavior_repair",), ("general_repair",)]
    assert all(spec.path.exists() for spec in pool)
    artifact = json.loads((generation_root / "generated_pool" / "generated_pool.json").read_text(encoding="utf-8"))
    assert artifact["task_id"] == "csv_report"
    assert [row["label"] for row in artifact["generated_pool"]] == ["candidate_001", "candidate_004"]
    assert artifact["generated_pool"][0]["generator"] == "llm_live"


def test_run_benchmark_suite_uses_generated_pool_once_per_task_and_marks_summary(tmp_path, monkeypatch):
    from gategrpo.benchmark import run_benchmark_suite

    generation_calls: list[Path] = []
    baseline_calls: list[tuple[str, tuple[str, ...], str]] = []

    generated_pool = [
        CandidateSpec(
            path=tmp_path / "llm_live_001.patch",
            label="live_001",
            compatible_routes=("behavior_repair",),
            generator="llm_live",
            intent="visible_tests",
        ),
        CandidateSpec(
            path=tmp_path / "llm_live_002.patch",
            label="live_002",
            compatible_routes=("general_repair",),
            generator="llm_live",
            intent="promoted",
        ),
    ]

    def fake_candidate_generator(task_dir: Path, task_run_dir: Path):
        generation_calls.append(task_dir)
        assert task_run_dir.name == load_task(task_dir).task_id
        return generated_pool

    def fake_run_baseline(baseline, task_dir, candidate_files, run_dir, budget):
        baseline_calls.append((baseline, tuple(spec.generator for spec in candidate_files), run_dir.name))
        assert candidate_files == generated_pool
        return {
            "baseline": baseline,
            "status": "promoted",
            "stop_reason": "promoted",
            "attempts": 1,
            "failure_types": [],
            "evidence_packets": 0,
            "selection_reasons": ["ordered_baseline"],
            "trace_path": str(run_dir),
            "archive_path": None,
        }

    monkeypatch.setattr("gategrpo.benchmark._run_baseline", fake_run_baseline)

    payload = run_benchmark_suite(
        SMOKE_SUITE,
        tmp_path / "benchmark",
        baselines=("single_shot", "full_gategrpo"),
        candidate_generator=fake_candidate_generator,
    )

    assert len(generation_calls) == 4
    assert len(set(generation_calls)) == 4
    assert payload["candidate_generator"] == "llm_live"
    assert payload["summary"]["full_gategrpo"]["candidate_generator"] == "llm_live"
    assert payload["results"][0]["candidate_generator"] == "llm_live"
    assert all(call[1] == ("llm_live", "llm_live") for call in baseline_calls)
    assert payload["summary"]["single_shot"]["solve_at_budget"] == 1.0
    assert payload["summary"]["full_gategrpo"]["solve_at_budget"] == 1.0



def test_controller_diversification_suite_strictly_lifts_full_gategrpo(tmp_path):
    from gategrpo.benchmark import run_benchmark_suite

    suite_path = ROOT / "benchmarks" / "controller_diversification_suite.json"
    payload = run_benchmark_suite(
        suite_path,
        tmp_path / "controller_diversification",
        repetitions=1,
        budget=SearchBudget(max_candidates=3, max_repeated_failures=5),
    )

    summary = payload["summary"]
    assert summary["full_gategrpo"]["solve_at_budget"] == 1.0
    for baseline in ("single_shot", "linear_retry", "clean_context_review", "evidence_aware_review", "archive_no_routing"):
        assert summary[baseline]["solve_at_budget"] == 0.0


def test_run_llm_benchmark_suite_passes_provider_to_run_llm_search(tmp_path, monkeypatch):
    seen_kwargs: list[dict] = []

    monkeypatch.setattr("gategrpo.benchmark.build_llm_client", lambda *a, **k: SimpleNamespace(model="fake-llm"))

    def fake_run_llm_search(task_dir, run_dir, budget, client=None, **kwargs):
        seen_kwargs.append(kwargs)
        task = load_task(task_dir)
        return SearchResult(
            task_id=task.task_id,
            status="promoted",
            stop_reason="promoted",
            trace_path=run_dir / "trace.jsonl",
            archive_path=run_dir / "archive.json",
            records=[SimpleNamespace(failure_type=None)],
            total_tokens=10,
            prompt_tokens=6,
            completion_tokens=4,
            token_source="provider_usage",
            cost=0.01,
            currency="USD",
        )

    monkeypatch.setattr("gategrpo.benchmark.run_llm_search", fake_run_llm_search)

    payload = run_llm_benchmark_suite(
        SMOKE_SUITE,
        tmp_path / "llm_benchmark",
        provider="vllm",
        temperature=0.2,
        seed=7,
        repetitions=1,
    )

    assert all(kwargs["provider"] == "vllm" for kwargs in seen_kwargs)
    assert all(kwargs["temperature"] == 0.2 for kwargs in seen_kwargs)
    assert all(kwargs["seed"] == 7 for kwargs in seen_kwargs)
    assert payload["summary"]["all_tasks"]["token_source"] == "provider_usage"
    assert payload["summary"]["all_tasks"]["total_prompt_tokens"] == 6 * 4
    assert payload["summary"]["all_tasks"]["total_completion_tokens"] == 4 * 4
    assert payload["summary"]["all_tasks"]["total_cost"] == 0.04
    assert payload["summary"]["all_tasks"]["currency"] == "USD"


def test_run_llm_model_matrix_merges_models_into_comparison(tmp_path, monkeypatch):
    calls: list[dict] = []

    def fake_run_llm_benchmark_suite(*, suite_path, run_dir, provider, model, **kwargs):
        calls.append({"provider": provider, "model": model, "run_dir": run_dir})
        run_dir.mkdir(parents=True, exist_ok=True)
        solved = 3 if model == "model-a" else 2
        return {
            "suite": "smoke",
            "model": model,
            "provider": provider,
            "results": [],
            "summary": {
                "task_x": {"runs": 4, "solved": solved, "solve_at_budget": solved / 4},
                "all_tasks": {
                    "runs": 4,
                    "solved": solved,
                    "solve_at_budget": solved / 4,
                    "solve_at_budget_ci95": (0.1, 0.9),
                    "avg_attempts": 1.5,
                    "avg_tokens": 100.0,
                    "token_source": "provider_usage",
                    "avg_wall_seconds": 0.2,
                },
            },
        }

    monkeypatch.setattr("gategrpo.benchmark.run_llm_benchmark_suite", fake_run_llm_benchmark_suite)

    result = run_llm_model_matrix(
        SMOKE_SUITE,
        tmp_path / "matrix",
        model_specs=[
            {"id": "model-a", "provider": "vllm", "model": "model-a"},
            {"id": "model-b", "provider": "vllm", "model": "model-b"},
        ],
        repetitions=4,
    )

    assert [call["model"] for call in calls] == ["model-a", "model-b"]
    assert [entry["model"] for entry in result["matrix"]] == ["model-a", "model-b"]
    assert result["matrix"][0]["overall"]["solve_at_budget"] == 0.75
    assert result["matrix"][1]["overall"]["solve_at_budget"] == 0.5
    assert (tmp_path / "matrix" / "model_matrix_results.json").exists()
    matrix_md = (tmp_path / "matrix" / "model_matrix_summary.md").read_text(encoding="utf-8")
    assert "model-a" in matrix_md and "model-b" in matrix_md

    report_path = tmp_path / "matrix" / "model_matrix_report.html"
    assert report_path.exists()
    report_html = report_path.read_text(encoding="utf-8")
    assert "Model comparison" in report_html
    assert "Per-task solve@budget" in report_html
    assert "model-a" in report_html and "model-b" in report_html
    assert "task_x" in report_html
    # best model per task highlighted
    assert "td.best" in report_html and "class='best'" in report_html


def test_llm_task_summary_reports_variance_and_currency():
    rows = [
        {"promoted": True, "attempts": 1, "total_tokens": 100, "wall_seconds": 2.0,
         "prompt_tokens": 60, "completion_tokens": 40, "token_source": "provider_usage",
         "cost": 0.10, "currency": "EUR"},
        {"promoted": False, "attempts": 2, "total_tokens": 300, "wall_seconds": 6.0,
         "prompt_tokens": 180, "completion_tokens": 120, "token_source": "provider_usage",
         "cost": 0.30, "currency": "EUR"},
    ]

    summary = _llm_task_summary(rows)

    assert summary["runs"] == 2
    assert summary["tokens_stdev"] > 0.0
    assert summary["wall_seconds_stdev"] > 0.0
    assert summary["total_cost"] == 0.4
    assert summary["currency"] == "EUR"


def test_organic_diversification_suite_strictly_lifts_full_gategrpo(tmp_path):
    from gategrpo.benchmark import run_benchmark_suite

    suite_path = ROOT / "benchmarks" / "organic_diversification_suite.json"
    payload = run_benchmark_suite(
        suite_path,
        tmp_path / "organic_diversification",
        repetitions=1,
        budget=SearchBudget(max_candidates=3, max_repeated_failures=5),
    )

    summary = payload["summary"]
    assert summary["full_gategrpo"]["solve_at_budget"] == 1.0
    for baseline in ("single_shot", "linear_retry", "clean_context_review", "evidence_aware_review", "archive_no_routing"):
        assert summary[baseline]["solve_at_budget"] == 0.0
