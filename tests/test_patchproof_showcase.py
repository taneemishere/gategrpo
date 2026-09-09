from pathlib import Path

from patchproof.controller import run_search
from patchproof.models import SearchBudget
from patchproof.report import generate_showcase_report


ROOT = Path(__file__).resolve().parents[1]
SMOKE_TASK = ROOT / "tasks" / "smoke_markdown_parser"


def test_generate_showcase_report_renders_expected_sections(tmp_path):
    run_search(SMOKE_TASK, tmp_path / "showcase")

    output = generate_showcase_report(tmp_path / "showcase", tmp_path / "showcase_report.html")
    html = output.read_text(encoding="utf-8")

    assert "Run Summary" in html
    assert "Problem" in html
    assert "Attempt Timeline" in html
    assert "Evidence &amp; Routing" in html
    assert "Gate Results" in html
    assert "Promoted Patch" in html
    assert "Artifacts" in html
    assert "Benchmark summary" not in html
    assert "solve@budget" not in html


def test_generate_showcase_report_shows_no_promotion_path(tmp_path):
    run_search(
        SMOKE_TASK,
        tmp_path / "no_promote",
        budget=SearchBudget(max_candidates=1, max_repeated_failures=1),
        candidate_files=[SMOKE_TASK / "candidate_visible_fail.patch"],
    )

    output = generate_showcase_report(tmp_path / "no_promote", tmp_path / "no_promote_report.html")
    html = output.read_text(encoding="utf-8")

    assert "No candidate was promoted" in html
    assert "last attempt" in html.lower()
