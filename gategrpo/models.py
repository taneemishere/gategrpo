from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TaskConfig:
    task_id: str
    name: str
    workspace: Path
    instructions: Path
    allowed_paths: tuple[str, ...]
    visible_tests: tuple[str, ...]
    regression_tests: tuple[str, ...]
    default_patch: Path | None = None
    candidate_patches: tuple[Path, ...] = ()


@dataclass(frozen=True)
class CandidateSpec:
    path: Path
    label: str
    compatible_routes: tuple[str, ...] = ()
    generator: str = "static_pool"
    intent: str = "unspecified"


@dataclass(frozen=True)
class CommandResult:
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    duration_seconds: float

    @property
    def passed(self) -> bool:
        return self.returncode == 0


@dataclass
class GateResult:
    name: str
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)
    fingerprint: str | None = None


@dataclass
class RunResult:
    task_id: str
    status: str
    workspace: Path
    trace_path: Path
    gates: list[GateResult]
    touched_files: tuple[str, ...] = ()
    duration_seconds: float = 0.0

    @property
    def passed(self) -> bool:
        return self.status == "promoted"


@dataclass(frozen=True)
class EvidencePacket:
    evidence_id: str
    gate: str
    failure_type: str
    summary: str
    details: dict[str, Any] = field(default_factory=dict)
    packet_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class CandidateRecord:
    candidate_id: str
    patch_file: str
    status: str
    route: str
    failure_type: str | None
    score: float
    gates: tuple[dict[str, Any], ...]
    parent_id: str | None = None
    generation: int = 0
    touched_files: tuple[str, ...] = ()
    candidate_label: str = ""
    candidate_intent: str = ""
    compatible_routes: tuple[str, ...] = ()
    selection_reason: str = ""
    rationale: str = ""
    verifier_results: tuple[dict[str, Any], ...] = ()
    failure_fingerprint: str | None = None
    token_estimate: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    token_source: str = "estimate"
    metadata_source: str = "controller"
    duration_seconds: float = 0.0
    evidence: EvidencePacket | None = None


@dataclass(frozen=True)
class SearchBudget:
    max_candidates: int = 2
    max_repeated_failures: int = 2
    max_wall_seconds: float | None = 300.0
    max_writer_tokens: int | None = None
    max_critic_tokens: int | None = None


@dataclass
class SearchResult:
    task_id: str
    status: str
    stop_reason: str
    trace_path: Path
    archive_path: Path
    records: list[CandidateRecord]
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    token_source: str = "estimate"
    cost: float | None = None
    currency: str | None = None
    wall_seconds: float = 0.0

    @property
    def passed(self) -> bool:
        return self.status == "promoted"
