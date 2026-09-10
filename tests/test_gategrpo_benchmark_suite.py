import json
from pathlib import Path

from gategrpo.benchmark import run_benchmark_suite
from gategrpo.models import SearchBudget


ROOT = Path(__file__).resolve().parents[1]


def test_benchmark_suite_writes_summary_outputs(tmp_path):
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(
        json.dumps(
            {
                "name": "test_suite",
                "cases": [
                    {
                        "id": "needs_repair",
                        "task_dir": str(ROOT / "tasks" / "smoke_markdown_parser"),
                        "candidate_patches": ["candidate_visible_fail.patch", "candidate.patch"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = run_benchmark_suite(
        suite_path=suite_path,
        run_dir=tmp_path / "benchmark",
        baselines=("single_shot", "full_gategrpo"),
        budget=SearchBudget(max_candidates=2, max_repeated_failures=2),
    )

    assert payload["summary"]["single_shot"]["solve_at_budget"] == 0.0
    assert payload["summary"]["full_gategrpo"]["solve_at_budget"] == 1.0
    assert payload["summary"]["full_gategrpo"]["evidence_packets_admitted"] == 1
    assert (tmp_path / "benchmark" / "benchmark_results.json").exists()
    assert (tmp_path / "benchmark" / "benchmark_summary.md").exists()


def test_evidence_aware_review_uses_bounded_evidence_and_metadata(tmp_path):
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(
        json.dumps(
            {
                "name": "route_metadata_suite",
                "cases": [
                    {
                        "id": "visible_route_decoy",
                        "task_dir": str(ROOT / "tasks" / "smoke_markdown_parser"),
                        "candidate_patches": [
                            {
                                "patch": "candidate_visible_fail.patch",
                                "label": "first_pass",
                                "intent": "first_pass_attempt",
                                "compatible_routes": [],
                            },
                            {
                                "patch": "candidate_scope_fail.patch",
                                "label": "scope_decoy",
                                "intent": "wrong_route_attempt",
                                "compatible_routes": ["scope_repair"],
                            },
                            {
                                "patch": "candidate.patch",
                                "label": "behavior_repair",
                                "intent": "complete_spec_repair",
                                "compatible_routes": ["behavior_repair"],
                            },
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = run_benchmark_suite(
        suite_path=suite_path,
        run_dir=tmp_path / "benchmark",
        baselines=("linear_retry", "evidence_aware_review", "full_gategrpo"),
        budget=SearchBudget(max_candidates=2, max_repeated_failures=2),
    )

    assert payload["summary"]["linear_retry"]["solve_at_budget"] == 0.0
    assert payload["summary"]["evidence_aware_review"]["solve_at_budget"] == 1.0
    assert payload["summary"]["evidence_aware_review"]["metadata_route_matches"] == 1
    assert payload["summary"]["evidence_aware_review"]["evidence_packets_admitted"] == 1
    assert payload["summary"]["full_gategrpo"]["solve_at_budget"] == 1.0
