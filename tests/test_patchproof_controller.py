import json
from pathlib import Path

from patchproof.controller import run_search
from patchproof.models import CandidateSpec, SearchBudget


ROOT = Path(__file__).resolve().parents[1]
SMOKE_TASK = ROOT / "tasks" / "smoke_markdown_parser"


def test_search_routes_from_failed_candidate_to_promoted_candidate(tmp_path):
    result = run_search(SMOKE_TASK, tmp_path / "search")

    assert result.status == "promoted"
    assert result.stop_reason == "promoted"
    assert [record.status for record in result.records] == ["rejected", "promoted"]
    assert result.records[0].failure_type == "visible_tests"
    assert result.records[0].evidence is not None
    assert result.records[0].evidence.summary == "visible developer tests failed"
    assert result.records[1].route == "behavior_repair"
    assert result.archive_path.exists()

    archive = json.loads(result.archive_path.read_text(encoding="utf-8"))
    assert [record["candidate_id"] for record in archive["records"]] == ["candidate_001", "candidate_002"]


def test_search_stops_after_repeated_failure_limit(tmp_path):
    bad_patch = SMOKE_TASK / "candidate_visible_fail.patch"
    result = run_search(
        SMOKE_TASK,
        tmp_path / "search",
        budget=SearchBudget(max_candidates=3, max_repeated_failures=2),
        candidate_files=[bad_patch, bad_patch],
    )

    assert result.status == "stopped"
    assert result.stop_reason == "repeated_failure_limit"
    assert len(result.records) == 2
    assert all(record.failure_type == "visible_tests" for record in result.records)


def test_route_selection_uses_candidate_metadata_not_filename_heuristics(tmp_path):
    candidates = [
        CandidateSpec(
            path=SMOKE_TASK / "candidate_visible_fail.patch",
            label="first_pass",
            intent="first_pass_attempt",
        ),
        CandidateSpec(
            path=SMOKE_TASK / "candidate_scope_fail.patch",
            label="scope_decoy",
            intent="irrelevant_scope_attempt",
            compatible_routes=("scope_repair",),
        ),
        CandidateSpec(
            path=SMOKE_TASK / "candidate.patch",
            label="metadata_selected_repair",
            intent="complete_spec_repair",
            compatible_routes=("behavior_repair",),
        ),
    ]

    result = run_search(
        SMOKE_TASK,
        tmp_path / "search",
        budget=SearchBudget(max_candidates=2, max_repeated_failures=2),
        candidate_files=candidates,
    )

    assert result.status == "promoted"
    assert [record.candidate_label for record in result.records] == ["first_pass", "metadata_selected_repair"]
    assert result.records[1].selection_reason == "metadata_route_match:behavior_repair"
