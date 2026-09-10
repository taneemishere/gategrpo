from pathlib import Path

import json

from gategrpo.archive import CandidateArchive, build_run_metadata
from gategrpo.models import CandidateRecord, SearchBudget
from gategrpo.task import load_task


ROOT = Path(__file__).resolve().parents[1]
SMOKE_TASK = ROOT / "tasks" / "smoke_markdown_parser"


def test_candidate_archive_save_without_run_header_writes_records_only(tmp_path):
    archive = CandidateArchive()
    archive.add(
        CandidateRecord(
            candidate_id="candidate_001",
            patch_file="candidate.patch",
            status="promoted",
            route="initial_repair",
            failure_type=None,
            score=1.0,
            gates=(),
        )
    )

    output = tmp_path / "candidate_archive.json"
    archive.save(output)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert "run" not in payload
    assert len(payload["records"]) == 1
    assert payload["records"][0]["candidate_id"] == "candidate_001"


def test_candidate_archive_save_with_run_header_includes_metadata(tmp_path):
    task = load_task(SMOKE_TASK)
    archive = CandidateArchive()
    archive.add(
        CandidateRecord(
            candidate_id="candidate_001",
            patch_file="candidate.patch",
            status="promoted",
            route="initial_repair",
            failure_type=None,
            score=1.0,
            gates=(),
        )
    )

    output = tmp_path / "candidate_archive.json"
    archive.save(
        output,
        run=build_run_metadata(
            task,
            archive.records,
            status="promoted",
            stop_reason="promoted",
            total_tokens=17,
            wall_seconds=1.5,
            budget=SearchBudget(max_candidates=10, max_repeated_failures=10),
            model="test-model",
            provider="openrouter",
            temperature=0.9,
        ),
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["run"]["task_id"] == task.task_id
    assert payload["run"]["model"] == "test-model"
    assert payload["run"]["budget"]["max_candidates"] == 10
    assert payload["run"]["repair_path"]
    assert len(payload["records"]) == 1
