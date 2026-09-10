from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from .task import load_task


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class InitTaskResult:
    task_dir: Path
    task_json: Path
    task_id: Optional[str]
    task_name: Optional[str]
    created: bool
    validated_only: bool
    checks: tuple[CheckResult, ...]
    notes: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)


def init_task(
    task_dir: Path,
    repo: Optional[Path] = None,
    instructions: Optional[Path] = None,
    allowed_paths: Sequence[str] = (),
    visible_tests: Optional[Path] = None,
    regression_tests: Optional[Path] = None,
    name: Optional[str] = None,
    validate: bool = False,
) -> InitTaskResult:
    task_dir = task_dir.resolve()
    task_json = task_dir / "task.json"
    existing_non_empty = task_dir.exists() and task_dir.is_dir() and any(task_dir.iterdir())
    if validate or existing_non_empty:
        notes = ("Existing non-empty task directory detected; running validate-only.",) if existing_non_empty and not validate else ()
        return validate_task(task_dir, notes=notes)

    if repo is None or instructions is None:
        raise ValueError("repo and instructions are required when creating a task")
    if not repo.exists() or not repo.is_dir():
        raise ValueError(f"repo directory does not exist: {repo}")
    if not instructions.exists() or not instructions.is_file():
        raise ValueError(f"instructions file does not exist: {instructions}")
    if not allowed_paths:
        raise ValueError("at least one --allowed-path is required")
    if visible_tests is not None and (not visible_tests.exists() or not visible_tests.is_dir()):
        raise ValueError(f"test directory does not exist: {visible_tests}")
    if regression_tests is not None and (not regression_tests.exists() or not regression_tests.is_dir()):
        raise ValueError(f"test directory does not exist: {regression_tests}")

    task_dir.mkdir(parents=True, exist_ok=True)
    task_dir_repo = task_dir / "repo"
    _copy_repo(repo, task_dir_repo)
    _copy_file(instructions, task_dir / "instructions.md")

    for allowed_path in allowed_paths:
        _ensure_allowed_path(task_dir_repo, allowed_path)

    notes: list[str] = []
    visible_test_path = _prepare_tests(task_dir_repo, visible_tests, "tests/visible")
    regression_test_path = _prepare_tests(task_dir_repo, regression_tests, "tests/regression")
    if visible_tests is None or regression_tests is None:
        notes.append("Placeholder visible/regression tests were scaffolded; replace them with real tests before using the task for evaluation.")

    task_id = _slugify(task_dir.name)
    task_name = name or _title_from_slug(task_dir.name)
    task_json.write_text(
        json.dumps(
            {
                "id": task_id,
                "name": task_name,
                "workspace": "repo",
                "instructions": "instructions.md",
                "allowed_paths": list(allowed_paths),
                "visible_tests": [visible_test_path],
                "regression_tests": [regression_test_path],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return validate_task(task_dir, created=True, notes=tuple(notes))


def validate_task(task_dir: Path, created: bool = False, notes: Sequence[str] = ()) -> InitTaskResult:
    task_dir = task_dir.resolve()
    task_json = task_dir / "task.json"
    checks: list[CheckResult] = []
    task_id: str | None = None
    task_name: str | None = None
    task = None

    if task_json.exists():
        try:
            task = load_task(task_dir)
        except Exception as exc:
            checks.append(CheckResult("task.json exists and parses", False, str(exc)))
        else:
            task_id = task.task_id
            task_name = task.name
            checks.append(CheckResult("task.json exists and parses", True, f"loaded as {task.task_id}"))
    else:
        checks.append(CheckResult("task.json exists and parses", False, "task.json missing"))

    if task is not None:
        workspace = task.workspace
        workspace_ok = workspace.exists() and workspace.is_dir()
        checks.append(CheckResult("workspace/repo dir exists", workspace_ok, str(workspace)))

        for allowed_path in task.allowed_paths:
            allowed = workspace / allowed_path
            checks.append(
                CheckResult(
                    f"allowed path exists: {allowed_path}",
                    allowed.exists(),
                    str(allowed),
                )
            )

        for visible_path in task.visible_tests:
            checks.append(_dir_check(f"visible tests dir exists: {visible_path}", workspace / visible_path))
        for regression_path in task.regression_tests:
            checks.append(_dir_check(f"regression tests dir exists: {regression_path}", workspace / regression_path))

        instructions = task.instructions
        if instructions.exists() and instructions.is_file():
            text = instructions.read_text(encoding="utf-8").strip()
            passed = bool(text)
            detail = "non-empty" if passed else "empty"
        else:
            passed = False
            detail = "missing"
        checks.append(CheckResult("instructions file exists and is non-empty", passed, detail))
    else:
        checks.extend(
            [
                CheckResult("workspace/repo dir exists", False, "task.json unavailable"),
                CheckResult("visible tests dir exists", False, "task.json unavailable"),
                CheckResult("regression tests dir exists", False, "task.json unavailable"),
                CheckResult("instructions file exists and is non-empty", False, "task.json unavailable"),
            ]
        )

    return InitTaskResult(
        task_dir=task_dir,
        task_json=task_json,
        task_id=task_id,
        task_name=task_name,
        created=created,
        validated_only=not created,
        checks=tuple(checks),
        notes=tuple(notes),
    )


def _copy_repo(src: Path, dest: Path) -> None:
    shutil.copytree(src, dest)


def _copy_file(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def _prepare_tests(workspace: Path, source: Optional[Path], relative_path: str) -> str:
    target = workspace / relative_path
    if source is None:
        target.mkdir(parents=True, exist_ok=True)
        placeholder = target / "test_placeholder.py"
        placeholder.write_text("def test_placeholder():\n    assert True\n", encoding="utf-8")
    else:
        shutil.copytree(source, target)
    return relative_path


def _ensure_allowed_path(repo: Path, allowed_path: str) -> None:
    normalized = Path(allowed_path)
    if normalized.is_absolute() or ".." in normalized.parts:
        raise ValueError(f"allowed path must be relative to repo: {allowed_path}")
    candidate = repo / normalized
    if not candidate.exists():
        raise ValueError(f"allowed path does not exist in repo: {allowed_path}")


def _dir_check(name: str, path: Path) -> CheckResult:
    return CheckResult(name, path.exists() and path.is_dir(), str(path))


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "task"


def _title_from_slug(value: str) -> str:
    return " ".join(part for part in re.split(r"[-_]+", value) if part).strip().title() or "Task"
