from __future__ import annotations

import json
from pathlib import Path

from .models import TaskConfig


def load_task(task_dir: Path) -> TaskConfig:
    config_path = task_dir / "task.json"
    with config_path.open(encoding="utf-8") as handle:
        raw = json.load(handle)

    default_patch = raw.get("default_patch")
    candidate_patches = raw.get("candidate_patches", [])
    return TaskConfig(
        task_id=str(raw["id"]),
        name=str(raw["name"]),
        workspace=task_dir / str(raw["workspace"]),
        instructions=task_dir / str(raw["instructions"]),
        allowed_paths=tuple(str(path) for path in raw.get("allowed_paths", [])),
        visible_tests=tuple(str(path) for path in raw.get("visible_tests", [])),
        regression_tests=tuple(str(path) for path in raw.get("regression_tests", [])),
        default_patch=(task_dir / str(default_patch)) if default_patch else None,
        candidate_patches=tuple(task_dir / str(path) for path in candidate_patches),
    )
