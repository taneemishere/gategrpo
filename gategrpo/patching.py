from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from .models import CommandResult


def parse_touched_files(patch_text: str) -> list[str]:
    touched: list[str] = []
    for line in patch_text.splitlines():
        if not line.startswith("+++ "):
            continue
        path = line[4:].strip()
        if path == "/dev/null":
            continue
        if path.startswith("b/"):
            path = path[2:]
        if path and path not in touched:
            touched.append(path)
    return touched


def run_command(command: list[str], cwd: Path, timeout_seconds: int = 60) -> CommandResult:
    started = time.monotonic()
    env = os.environ.copy()
    env["GIT_CEILING_DIRECTORIES"] = str(cwd.resolve().parent)
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    return CommandResult(
        command=tuple(command),
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        duration_seconds=time.monotonic() - started,
    )


def check_patch_applies(workspace: Path, patch_path: Path) -> CommandResult:
    return run_command(["git", "apply", "--check", str(patch_path)], cwd=workspace)


def apply_patch_file(workspace: Path, patch_path: Path) -> CommandResult:
    return run_command(["git", "apply", str(patch_path)], cwd=workspace)
