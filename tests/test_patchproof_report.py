import json
from pathlib import Path

import pytest

from patchproof.benchmark import run_benchmark_suite
from patchproof.models import SearchBudget
from patchproof.report import generate_html_report, generate_model_matrix_report
from patchproof.controller import run_search


ROOT = Path(__file__).resolve().parents[1]
SMOKE_TASK = ROOT / "tasks" / "smoke_markdown_parser"
SMOKE_SUITE = ROOT / "benchmarks" / "repair_routing_suite.json"


def test_generates_search_html_report(tmp_path):
    run_search(SMOKE_TASK, tmp_path / "search")

    output = generate_html_report(tmp_path / "search", tmp_path / "search_report.html")
    html = output.read_text(encoding="utf-8")

    assert "Search trace" in html
    assert "candidate_001" in html
    assert "visible developer tests failed" in html
    assert "Patch diff" in html


def test_generates_benchmark_html_report(tmp_path):
    run_benchmark_suite(
        SMOKE_SUITE,
        tmp_path / "benchmark",
        baselines=("single_shot", "full_patchproof"),
        budget=SearchBudget(max_candidates=2, max_repeated_failures=2),
    )

    output = generate_html_report(tmp_path / "benchmark", tmp_path / "benchmark_report.html")
    html = output.read_text(encoding="utf-8")

    assert "Benchmark summary: repair_routing_suite" in html
    assert "solve@budget" in html
    assert "full_patchproof" in html


def _write_matrix(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "suite": "repair_routing_suite",
        "matrix": [
            {
                "id": "qwen",
                "provider": "vllm",
                "model": "Qwen3-Coder",
                "overall": {
                    "runs": 4,
                    "solve_at_budget": 0.75,
                    "solve_at_budget_ci95": (0.3, 0.95),
                    "avg_attempts": 1.5,
                    "avg_tokens": 1200.0,
                    "tokens_stdev": 300.0,
                    "token_source": "provider_usage",
                    "avg_cost": 0.12,
                    "currency": "EUR",
                    "avg_wall_seconds": 4.2,
                    "wall_seconds_stdev": 1.1,
                },
                "per_task": {
                    "task_x": {"solve_at_budget": 1.0, "solve_at_budget_ci95": (0.5, 1.0)},
                    "task_y": {"solve_at_budget": 0.5, "solve_at_budget_ci95": (0.1, 0.9)},
                },
            },
            {
                "id": "gpt",
                "provider": "openrouter",
                "model": "gpt-oss",
                "overall": {
                    "runs": 4,
                    "solve_at_budget": 0.5,
                    "solve_at_budget_ci95": (0.2, 0.8),
                    "avg_attempts": 2.0,
                    "avg_tokens": 900.0,
                    "token_source": "estimate",
                    "avg_wall_seconds": 3.1,
                },
                "per_task": {
                    "task_x": {"solve_at_budget": 0.25, "solve_at_budget_ci95": (0.0, 0.6)},
                },
            },
        ],
    }
    (run_dir / "model_matrix_results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )


def test_generates_model_matrix_html_report(tmp_path):
    _write_matrix(tmp_path / "matrix")

    output = generate_model_matrix_report(tmp_path / "matrix", tmp_path / "matrix.html")
    html = output.read_text(encoding="utf-8")

    assert "Model comparison: repair_routing_suite" in html
    assert "Per-task solve@budget" in html
    # both models compared side by side
    assert "Qwen3-Coder" in html and "gpt-oss" in html
    # currency-aware cost and a missing-usage cost surface correctly
    assert "0.12 EUR" in html
    assert "n/a" in html  # gpt has no avg_cost, and no task_y result
    # variance rendered inline
    assert "1200.0 ± 300.0" in html
    # best-per-task highlight present for task_x (qwen wins 1.0 vs 0.25)
    assert "class='best'" in html


def test_model_matrix_report_requires_artifact(tmp_path):
    with pytest.raises(ValueError, match="model matrix artifact"):
        generate_model_matrix_report(tmp_path / "missing", tmp_path / "out.html")
