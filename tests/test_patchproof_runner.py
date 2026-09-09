import sys
from pathlib import Path

from patchproof import gates
from patchproof.patching import parse_touched_files
from patchproof.runner import run_task


ROOT = Path(__file__).resolve().parents[1]
SMOKE_TASK = ROOT / "tasks" / "smoke_markdown_parser"


def test_parse_touched_files_uses_new_paths_only():
    patch_text = """
diff --git a/parser.py b/parser.py
--- a/parser.py
+++ b/parser.py
@@ -1 +1 @@
-old
+new
"""

    assert parse_touched_files(patch_text) == ["parser.py"]


def test_smoke_task_promotes_default_candidate(tmp_path):
    result = run_task(SMOKE_TASK, tmp_path / "run")

    assert result.status == "promoted"
    assert result.trace_path.exists()
    assert [gate.name for gate in result.gates] == [
        "scope_guard",
        "patch_applies",
        "python_ast",
        "secret_scan",
        "visible_tests",
        "release_gate_regressions",
    ]
    assert all(gate.passed for gate in result.gates)


def test_scope_guard_rejects_out_of_scope_patch(tmp_path):
    patch_file = tmp_path / "bad.patch"
    patch_file.write_text(
        """diff --git a/spec.md b/spec.md
index 1111111..2222222 100644
--- a/spec.md
+++ b/spec.md
@@ -1 +1 @@
-# Markdown table parser spec
+# Changed spec
""",
        encoding="utf-8",
    )

    result = run_task(SMOKE_TASK, tmp_path / "run", patch_file=patch_file)

    assert result.status == "rejected"
    assert result.gates[0].name == "scope_guard"
    assert not result.gates[0].passed
    assert result.gates[0].details["blocked_files"] == ["spec.md"]


def test_run_with_env_turns_timeout_into_failed_result(monkeypatch, tmp_path):
    monkeypatch.setattr(gates, "TEST_TIMEOUT_SECONDS", 1)
    result = gates._run_with_env(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        tmp_path,
        gates._safe_test_env(),
    )

    assert result.returncode == 124
    assert not result.passed
    assert "timed out after 1 seconds" in result.stderr
    assert result.duration_seconds >= 0
