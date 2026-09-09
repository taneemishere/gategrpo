from pathlib import Path

import json

from patchproof.cli import main
from patchproof.init_task import init_task, validate_task
from patchproof.task import load_task


def _write_sample_repo(root: Path) -> tuple[Path, Path]:
    repo = root / "example_repo"
    repo.mkdir()
    (repo / "parser.py").write_text(
        "def parse_text(text: str) -> str:\n    return text.strip()\n",
        encoding="utf-8",
    )
    instructions = root / "instructions.md"
    instructions.write_text("# Example task\n\nRepair parser.py.\n", encoding="utf-8")
    return repo, instructions


def test_init_task_scaffolds_placeholder_tests_and_valid_task_json(tmp_path):
    repo, instructions = _write_sample_repo(tmp_path)
    task_dir = tmp_path / "my_task"

    result = init_task(
        task_dir,
        repo=repo,
        instructions=instructions,
        allowed_paths=("parser.py",),
    )

    assert result.passed
    assert result.created
    assert (task_dir / "repo" / "parser.py").exists()
    assert (task_dir / "instructions.md").exists()
    assert (task_dir / "repo" / "tests" / "visible" / "test_placeholder.py").exists()
    assert (task_dir / "repo" / "tests" / "regression" / "test_placeholder.py").exists()

    payload = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    assert payload["workspace"] == "repo"
    assert payload["instructions"] == "instructions.md"
    assert payload["allowed_paths"] == ["parser.py"]
    assert payload["visible_tests"] == ["tests/visible"]
    assert payload["regression_tests"] == ["tests/regression"]
    assert load_task(task_dir).task_id == "my-task"


def test_scaffolded_tests_live_in_workspace_and_are_gate_discoverable(tmp_path):
    repo, instructions = _write_sample_repo(tmp_path)
    visible = tmp_path / "src_visible"
    visible.mkdir()
    (visible / "test_ok.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    task_dir = tmp_path / "gate_task"

    init_task(
        task_dir,
        repo=repo,
        instructions=instructions,
        allowed_paths=("parser.py",),
        visible_tests=visible,
    )

    task = load_task(task_dir)
    # Tests must be copied inside the workspace so the gate (cwd=workspace) can find them.
    assert (task.workspace / "tests" / "visible" / "test_ok.py").exists()

    from patchproof.gates import pytest_gate

    gate = pytest_gate(task.workspace, tuple(task.visible_tests), "visible_tests")
    assert gate.passed
    assert not gate.details.get("skipped")
    assert "no tests ran" not in str(gate.details).lower()


def test_validate_task_passes_and_fails_on_breakage(tmp_path):
    repo, instructions = _write_sample_repo(tmp_path)
    task_dir = tmp_path / "validate_me"

    init_task(
        task_dir,
        repo=repo,
        instructions=instructions,
        allowed_paths=("parser.py",),
    )

    good = validate_task(task_dir)
    assert good.passed
    assert all(check.passed for check in good.checks)

    (task_dir / "repo" / "parser.py").unlink()
    broken = validate_task(task_dir)
    assert not broken.passed
    assert any(not check.passed and "allowed path exists" in check.name for check in broken.checks)


def test_init_task_existing_non_empty_directory_falls_back_to_validate_only(tmp_path):
    repo, instructions = _write_sample_repo(tmp_path)
    task_dir = tmp_path / "existing_task"
    task_dir.mkdir()
    marker = task_dir / "keep.txt"
    marker.write_text("keep me", encoding="utf-8")

    result = init_task(
        task_dir,
        repo=repo,
        instructions=instructions,
        allowed_paths=("parser.py",),
    )

    assert result.validated_only
    assert marker.read_text(encoding="utf-8") == "keep me"
    assert not (task_dir / "repo").exists()


def test_init_task_cli_validate_only_needs_no_creation_args(tmp_path, capsys):
    repo, instructions = _write_sample_repo(tmp_path)
    task_dir = tmp_path / "cli_validate"
    init_task(task_dir, repo=repo, instructions=instructions, allowed_paths=("parser.py",))

    exit_code = main(["init-task", str(task_dir), "--validate"])
    captured = capsys.readouterr().out

    assert exit_code == 0
    assert "Validated:" in captured
