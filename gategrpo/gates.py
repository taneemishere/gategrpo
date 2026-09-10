from __future__ import annotations

import ast
import os
import re
import sys
from pathlib import Path

from .models import CommandResult, GateResult
from .patching import run_command


SECRET_PATTERNS = {
    "openrouter_key": re.compile(r"sk-or-v1-[A-Za-z0-9]{20,}"),
    "openai_key": re.compile(r"sk-[A-Za-z0-9_-]{32,}"),
    "aws_access_key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "private_key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
}

TEST_TIMEOUT_SECONDS = 120


def scope_guard(touched_files: list[str], allowed_paths: tuple[str, ...]) -> GateResult:
    blocked = [
        path
        for path in touched_files
        if not any(path == allowed or path.startswith(f"{allowed.rstrip('/')}/") for allowed in allowed_paths)
    ]
    return GateResult(
        name="scope_guard",
        passed=not blocked,
        details={"touched_files": touched_files, "allowed_paths": list(allowed_paths), "blocked_files": blocked},
        fingerprint="scope_guard:" + ",".join(sorted(blocked)) if blocked else None,
    )


def python_ast_gate(workspace: Path, touched_files: list[str]) -> GateResult:
    checked: list[str] = []
    failures: list[dict[str, str | int]] = []
    for relative in touched_files:
        if not relative.endswith(".py"):
            continue
        path = workspace / relative
        if not path.exists():
            failures.append({"file": relative, "error": "file_missing_after_patch"})
            continue
        checked.append(relative)
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            failures.append(
                {
                    "file": relative,
                    "line": exc.lineno or 0,
                    "offset": exc.offset or 0,
                    "error": exc.msg,
                }
            )
    fingerprint = None
    if failures:
        first = failures[0]
        fingerprint = f"python_ast:{first.get('file')}:{first.get('line')}:{first.get('error')}"
    return GateResult(
        name="python_ast",
        passed=not failures,
        details={"checked": checked, "failures": failures},
        fingerprint=fingerprint,
    )


def secret_scan_gate(workspace: Path) -> GateResult:
    findings: list[dict[str, str]] = []
    for path in workspace.rglob("*"):
        if not path.is_file() or _is_ignored(path):
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        relative = path.relative_to(workspace).as_posix()
        for name, pattern in SECRET_PATTERNS.items():
            if pattern.search(content):
                findings.append({"file": relative, "pattern": name})
    fingerprint = None
    if findings:
        first = findings[0]
        fingerprint = f"secret_scan:{first.get('file')}:{first.get('pattern')}"
    return GateResult(name="secret_scan", passed=not findings, details={"findings": findings}, fingerprint=fingerprint)


def pytest_gate(workspace: Path, test_paths: tuple[str, ...], name: str) -> GateResult:
    if not test_paths:
        return GateResult(name=name, passed=True, details={"skipped": True, "reason": "no_tests_configured"})
    env = _safe_test_env()
    env["PYTHONPATH"] = str(workspace)
    command = [sys.executable, "-m", "pytest", "-q", *test_paths]
    if name == "release_gate_regressions":
        command = [sys.executable, "-m", "pytest", "-q", "--tb=no", "-rf", *test_paths]
    result = _run_with_env(command, workspace, env)
    expose_output = name == "visible_tests"
    failing_tests = _pytest_failed_tests(result.stdout) if name == "release_gate_regressions" and not result.passed else []
    details = {
        "command": list(result.command),
        "returncode": result.returncode,
        "duration_seconds": round(result.duration_seconds, 6),
    }
    if expose_output:
        details.update(
            {
                "stdout": result.stdout[-4000:],
                "stderr": result.stderr[-4000:],
            }
        )
    else:
        details.update(
            {
                "stdout": "" if result.passed else "withheld",
                "stderr": "" if result.passed else "withheld",
                "diagnostic": "release-gate output withheld from repair loop",
                "failing_tests": failing_tests,
                "failing_test_count": len(failing_tests),
            }
        )
    return GateResult(
        name=name,
        passed=result.passed,
        details=details,
        fingerprint=None if result.passed else _pytest_fingerprint(name, result.stdout, result.stderr),
    )


def _run_with_env(command: list[str], cwd: Path, env: dict[str, str]) -> CommandResult:
    import subprocess
    import time

    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=TEST_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = _normalize_timeout_output(exc.stdout)
        stderr = _normalize_timeout_output(exc.stderr)
        timeout_marker = f"timed out after {TEST_TIMEOUT_SECONDS} seconds"
        stderr = "\n".join(part for part in (stderr, timeout_marker) if part)
        return CommandResult(
            command=tuple(command),
            returncode=124,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=time.monotonic() - started,
        )
    return CommandResult(
        command=tuple(command),
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        duration_seconds=time.monotonic() - started,
    )


def _normalize_timeout_output(output: str | bytes | None) -> str:
    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    return output


def _is_ignored(path: Path) -> bool:
    ignored_parts = {".git", "__pycache__", ".pytest_cache"}
    return any(part in ignored_parts for part in path.parts)


def _safe_test_env() -> dict[str, str]:
    allowed = {
        "PATH",
        "HOME",
        "LANG",
        "LC_ALL",
        "PYTHONPATH",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "SSL_CERT_FILE",
        "REQUESTS_CA_BUNDLE",
    }
    env = {key: value for key, value in os.environ.items() if key in allowed}
    for key in list(env):
        upper = key.upper()
        if any(marker in upper for marker in ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")):
            env.pop(key, None)
    env["GATEGRPO_NETWORK_DISABLED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _pytest_fingerprint(name: str, stdout: str, stderr: str) -> str:
    text = f"{stdout}\n{stderr}"
    failed = re.search(r"FAILED\s+([^\s]+)", text)
    if failed:
        return f"{name}:{failed.group(1)}"
    error = re.search(r"(AssertionError|ValueError|TypeError|SyntaxError|ImportError|ModuleNotFoundError)", text)
    if error:
        return f"{name}:{error.group(1)}"
    return f"{name}:returncode_nonzero"


def _pytest_failed_tests(stdout: str) -> list[str]:
    return re.findall(r"(?m)^FAILED\s+(\S+)", stdout)
