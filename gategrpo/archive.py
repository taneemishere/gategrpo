from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

from .models import CandidateRecord, SearchBudget, TaskConfig


def build_run_metadata(
    task: TaskConfig,
    records: list[CandidateRecord],
    status: str,
    stop_reason: str,
    total_tokens: int,
    wall_seconds: float,
    budget: SearchBudget | None = None,
    model: str | None = None,
    provider: str | None = None,
    temperature: float | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    token_source: str | None = None,
    cost: float | None = None,
    currency: str | None = None,
    seed: int | None = None,
    reasoning_enabled: bool | None = None,
    endpoint: str | None = None,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "task_id": task.task_id,
        "task_name": task.name,
        "workspace": str(task.workspace),
        "allowed_paths": list(task.allowed_paths),
        "instructions_excerpt": _instructions_excerpt(task.instructions),
        "status": status,
        "stop_reason": stop_reason,
        "attempts": len(records),
        "total_tokens": total_tokens,
        "wall_seconds": wall_seconds,
        "model": model,
        "provider": provider,
        "budget": _budget_metadata(budget, temperature),
        "repair_path": _repair_path(records, stop_reason),
        "reproducibility": _reproducibility_metadata(
            budget=budget,
            model=model,
            provider=provider,
            endpoint=endpoint,
            temperature=temperature,
            reasoning_enabled=reasoning_enabled,
            seed=seed,
        ),
    }
    if prompt_tokens is not None:
        metadata["prompt_tokens"] = prompt_tokens
    if completion_tokens is not None:
        metadata["completion_tokens"] = completion_tokens
    if token_source is not None:
        metadata["token_source"] = token_source
    if cost is not None:
        metadata["cost"] = cost
        metadata["currency"] = currency or "USD"
    return metadata


def _reproducibility_metadata(
    *,
    budget: SearchBudget | None,
    model: str | None,
    provider: str | None,
    endpoint: str | None,
    temperature: float | None,
    reasoning_enabled: bool | None,
    seed: int | None,
) -> dict[str, object]:
    config = {
        "model": model,
        "provider": provider,
        "endpoint": endpoint,
        "temperature": temperature,
        "reasoning_enabled": reasoning_enabled,
        "seed": seed,
        "max_candidates": budget.max_candidates if budget is not None else None,
        "max_repeated_failures": (
            budget.max_repeated_failures if budget is not None else None
        ),
    }
    fingerprint = hashlib.sha256(
        json.dumps(config, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    return {
        "gategrpo_version": _gategrpo_version(),
        **config,
        "config_fingerprint": fingerprint,
    }


def _gategrpo_version() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    revision = sha.stdout.strip()
    return revision or "unknown"


class CandidateArchive:
    def __init__(self) -> None:
        self.records: list[CandidateRecord] = []

    def add(self, record: CandidateRecord) -> None:
        self.records.append(record)

    def repeated_failures(self, failure_key: str | None) -> int:
        if failure_key is None:
            return 0
        return sum(
            1
            for record in self.records
            if (record.failure_fingerprint or record.failure_type) == failure_key
        )

    def best_parent_id(self) -> str | None:
        if not self.records:
            return None
        return max(self.records, key=lambda record: record.score).candidate_id

    def next_generation(self) -> int:
        if not self.records:
            return 0
        return max(record.generation for record in self.records) + 1

    def save(self, path: Path, run: dict[str, object] | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, object] = {"records": [asdict(record) for record in self.records]}
        if run is not None:
            payload["run"] = run
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _budget_metadata(budget: SearchBudget | None, temperature: float | None) -> dict[str, object]:
    payload: dict[str, object] = {}
    if budget is not None:
        payload.update(
            {
                "max_candidates": budget.max_candidates,
                "max_repeated_failures": budget.max_repeated_failures,
            }
        )
        if budget.max_wall_seconds is not None:
            payload["max_wall_seconds"] = budget.max_wall_seconds
        if budget.max_writer_tokens is not None:
            payload["max_writer_tokens"] = budget.max_writer_tokens
        if budget.max_critic_tokens is not None:
            payload["max_critic_tokens"] = budget.max_critic_tokens
    if temperature is not None:
        payload["temperature"] = temperature
    return payload


def _instructions_excerpt(path: Path, limit: int = 400) -> str:
    text = path.read_text(encoding="utf-8").strip()
    excerpt = " ".join(text.split())
    if len(excerpt) > limit:
        return excerpt[: limit - 1].rstrip() + "…"
    return excerpt


def _repair_path(records: list[CandidateRecord], stop_reason: str) -> str:
    if not records:
        return stop_reason
    parts: list[str] = []
    first_failure = records[0].failure_type or "direct"
    parts.append(first_failure)
    for index, record in enumerate(records):
        parts.append(record.route)
        if record.status == "promoted":
            parts.append("promoted")
            break
        if index + 1 < len(records):
            next_failure = records[index + 1].failure_type
            if next_failure:
                parts.append(next_failure)
        else:
            parts.append(stop_reason)
    return " → ".join(parts)
