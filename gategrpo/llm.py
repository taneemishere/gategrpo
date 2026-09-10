from __future__ import annotations

import json
import difflib
import hashlib
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .archive import CandidateArchive, build_run_metadata
from .critic import KNOWN_ROUTES, CleanContextCritic
from .evidence import build_evidence_packet, classify_failure, context_packet_index
from .models import CandidateRecord, EvidencePacket, SearchBudget, SearchResult
from .pricing import estimate_cost, load_price_schedule, schedule_currency
from .runner import run_task
from .task import load_task
from .tracing import TraceWriter

OPENROUTER_MODEL = "openai/gpt-oss-120b:free"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
VLLM_DEFAULT_MODEL = "Qwen3-Coder-30B-A3B-Instruct"
VLLM_DEFAULT_BASE_URL = "http://localhost:8000/v1"
# Keep enough headroom for the model's completion while preserving the full transcript.
LLM_CHAT_HISTORY_PROMPT_TOKEN_CAP = 12000

# Optional trailing block the generator may emit to declare its own candidate
# metadata (intent and compatible repair routes). When present it is recorded as
# generator-derived metadata; when absent the controller derives metadata itself.
_GENERATOR_METADATA_INSTRUCTION = (
    "Optionally, after the SEARCH/REPLACE block(s), you may append exactly one fenced "
    "metadata block describing your own candidate. Use this format and nothing else after it: "
    "```gategrpo-metadata\\n{\"intent\": \"<short_intent>\", \"compatible_routes\": "
    "[\"behavior_repair\", \"regression_repair\"]}\\n```"
    " compatible_routes must be drawn from this set only: scope_repair, patch_format_repair, "
    "syntax_repair, safety_repair, behavior_repair, regression_repair, general_repair. "
    "Any route outside this set is ignored."
)


@dataclass(frozen=True)
class LLMResponse:
    content: str
    reasoning_details: object | None
    raw_message: dict[str, object]
    usage: dict[str, object] | None = None


@dataclass(frozen=True)
class GeneratorMetadata:
    intent: str | None
    compatible_routes: tuple[str, ...]


@dataclass(frozen=True)
class SearchReplaceBlock:
    path: str
    search_text: str
    replace_text: str


@dataclass(frozen=True)
class SearchReplacePatchResult:
    patch_text: str
    all_blocks_matched: bool
    match_status: str
    used_whitespace_tolerance: bool
    matched_paths: tuple[str, ...]
    whitespace_tolerant_paths: tuple[str, ...]
    failed_paths: tuple[str, ...]


class CachedLLMClient:
    def __init__(
        self,
        inner: OpenRouterClient | OpenAICompatibleClient,
        cache_dir: str | Path,
        cache_context: dict[str, object] | None = None,
    ) -> None:
        self.inner = inner
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_context = dict(cache_context) if cache_context else {}

    @property
    def model(self) -> str:
        return self.inner.model

    @property
    def reasoning_enabled(self) -> bool:
        return self.inner.reasoning_enabled

    @property
    def temperature(self) -> float | None:
        return self.inner.temperature

    @property
    def seed(self) -> int | None:
        return self.inner.seed

    @property
    def url(self) -> str:
        return self.inner.url

    def chat(self, messages: list[dict[str, object]]) -> LLMResponse:
        cache_path = self.cache_dir / self._cache_key(messages)
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            raw_message = cached.get("raw_message")
            usage = cached.get("usage")
            return LLMResponse(
                content=str(cached.get("content", "")),
                reasoning_details=cached.get("reasoning_details"),
                raw_message=raw_message if isinstance(raw_message, dict) else {},
                usage=usage if isinstance(usage, dict) else None,
            )

        response = self.inner.chat(messages)
        cache_path.write_text(
            json.dumps(
                {
                    "content": response.content,
                    "reasoning_details": response.reasoning_details,
                    "raw_message": response.raw_message,
                    "usage": response.usage,
                },
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
            ),
            encoding="utf-8",
        )
        return response

    def _cache_key(self, messages: list[dict[str, object]]) -> str:
        payload = {
            "version": 2,
            "model": self.inner.model,
            "reasoning_enabled": self.inner.reasoning_enabled,
            "context": self.cache_context,
            "messages": messages,
        }
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        return f"{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}.json"


class OpenRouterClient:
    def __init__(
        self,
        api_key: str,
        model: str = OPENROUTER_MODEL,
        url: str = OPENROUTER_URL,
        reasoning_enabled: bool = True,
        temperature: float | None = None,
        seed: int | None = None,
        opener: Callable[..., object] = urllib.request.urlopen,
    ) -> None:
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for --llm mode")
        self.api_key = api_key
        self.model = model
        self.url = url
        self.reasoning_enabled = reasoning_enabled
        self.temperature = temperature
        self.seed = seed
        self.opener = opener

    @classmethod
    def from_env(
        cls,
        model: str = OPENROUTER_MODEL,
        reasoning_enabled: bool = True,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> OpenRouterClient:
        return cls(
            os.environ.get("OPENROUTER_API_KEY", ""),
            model=model,
            reasoning_enabled=reasoning_enabled,
            temperature=temperature,
            seed=seed,
        )

    def chat(self, messages: list[dict[str, object]]) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": messages,
        }
        if self.reasoning_enabled:
            payload["reasoning"] = {"enabled": True}
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.seed is not None:
            payload["seed"] = self.seed
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with self.opener(request, timeout=180) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raw = _read_error_json(exc)
            raise RuntimeError(f"OpenRouter request failed with HTTP {exc.code}: {_summarize_openrouter_payload(raw)}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"OpenRouter request failed: {exc.reason}") from exc

        message = _extract_message(raw)
        content = message.get("content")
        return LLMResponse(
            content=content if isinstance(content, str) else "",
            reasoning_details=message.get("reasoning_details"),
            raw_message=message,
            usage=_extract_usage(raw),
        )


class OpenAICompatibleClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        reasoning_enabled: bool = False,
        temperature: float | None = None,
        seed: int | None = None,
        opener: Callable[..., object] = urllib.request.urlopen,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.url = f"{self.base_url}/chat/completions"
        self.model = model
        self.api_key = api_key
        self.reasoning_enabled = reasoning_enabled
        self.temperature = temperature
        self.seed = seed
        self.opener = opener

    def chat(self, messages: list[dict[str, object]]) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 4096,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.seed is not None:
            payload["seed"] = self.seed
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with self.opener(request, timeout=180) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raw = _read_error_json(exc)
            raise RuntimeError(
                f"vLLM/OpenAI-compatible request to {self.base_url} failed with HTTP {exc.code}: "
                f"{_summarize_openrouter_payload(raw)}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"vLLM/OpenAI-compatible request to {self.base_url} failed: {exc.reason}"
            ) from exc

        message = _extract_message(raw)
        content = message.get("content")
        return LLMResponse(
            content=content if isinstance(content, str) else "",
            reasoning_details=message.get("reasoning_details"),
            raw_message=message,
            usage=_extract_usage(raw),
        )


def build_llm_client(
    provider: str,
    *,
    model: str | None,
    base_url: str | None,
    api_key: str | None,
    reasoning_enabled: bool,
    temperature: float | None = None,
    seed: int | None = None,
    cache_dir: str | Path | None = None,
) -> OpenRouterClient | OpenAICompatibleClient | CachedLLMClient:
    resolved_cache_dir = cache_dir if cache_dir is not None else (os.environ.get("GATEGRPO_LLM_CACHE_DIR") or None)
    if provider == "openrouter":
        client: OpenRouterClient | OpenAICompatibleClient = OpenRouterClient.from_env(
            model or OPENROUTER_MODEL,
            reasoning_enabled=reasoning_enabled,
            temperature=temperature,
            seed=seed,
        )
    elif provider in {"vllm", "openai"}:
        client = OpenAICompatibleClient(
            base_url=base_url or os.environ.get("VLLM_BASE_URL") or VLLM_DEFAULT_BASE_URL,
            model=model or VLLM_DEFAULT_MODEL,
            api_key=api_key,
            reasoning_enabled=reasoning_enabled,
            temperature=temperature,
            seed=seed,
        )
    else:
        raise RuntimeError("Unsupported LLM provider. Valid providers: openrouter, vllm, openai")
    if resolved_cache_dir is not None:
        cache_context = {
            "provider": provider,
            "endpoint": client.url,
            "temperature": client.temperature,
            "seed": client.seed,
        }
        return CachedLLMClient(client, resolved_cache_dir, cache_context=cache_context)
    return client


def run_llm_search(
    task_dir: Path,
    run_dir: Path,
    budget: SearchBudget | None = None,
    client: OpenRouterClient | OpenAICompatibleClient | CachedLLMClient | None = None,
    model: str | None = None,
    reasoning_enabled: bool = True,
    temperature: float | None = None,
    seed: int | None = None,
    provider: str = "openrouter",
    base_url: str | None = None,
    api_key: str | None = None,
    cache_dir: str | Path | None = None,
    price_schedule: dict[str, object] | str | Path | None = None,
) -> SearchResult:
    task = load_task(task_dir)
    budget = budget or SearchBudget(max_candidates=2, max_repeated_failures=2)
    resolved_price_schedule = (
        price_schedule if isinstance(price_schedule, dict) else load_price_schedule(price_schedule)
    )
    client = client or build_llm_client(
        provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
        reasoning_enabled=reasoning_enabled,
        temperature=temperature,
        seed=seed,
        cache_dir=cache_dir,
    )
    started = time.monotonic()
    run_dir.mkdir(parents=True, exist_ok=True)

    trace_path = run_dir / "controller_trace.jsonl"
    if trace_path.exists():
        trace_path.unlink()
    trace = TraceWriter(trace_path)
    archive = CandidateArchive()
    critic = CleanContextCritic()
    llm_dir = run_dir / "llm"
    llm_dir.mkdir(parents=True, exist_ok=True)

    messages: list[dict[str, object]] = [{"role": "user", "content": _initial_prompt(task_dir)}]
    writer_tokens = _estimate_text_tokens(str(messages[0]["content"]))
    usage_prompt_tokens = 0
    usage_completion_tokens = 0
    usage_reported = True
    trace.emit(
        "llm_search_started",
        task_id=task.task_id,
        model=client.model,
        reasoning_enabled=client.reasoning_enabled,
        max_candidates=budget.max_candidates,
        max_repeated_failures=budget.max_repeated_failures,
    )

    stop_reason = "candidate_budget_exhausted"
    for index in range(1, budget.max_candidates + 1):
        if _wall_budget_exhausted(started, budget):
            stop_reason = "wall_clock_budget_exhausted"
            trace.emit("llm_search_stopped", reason=stop_reason)
            break
        if budget.max_writer_tokens is not None and writer_tokens >= budget.max_writer_tokens:
            stop_reason = "writer_token_budget_exhausted"
            trace.emit("llm_search_stopped", reason=stop_reason, writer_tokens=writer_tokens)
            break
        candidate_id = f"llm_candidate_{index:03d}"
        route = "initial_repair" if not archive.records else critic.route_next(archive.records, archive.records[-1].evidence)
        _trim_chat_history(messages, LLM_CHAT_HISTORY_PROMPT_TOKEN_CAP)
        trace.emit("llm_candidate_requested", candidate_id=candidate_id, route=route, messages=len(messages))

        response = client.chat(messages)
        response_path = llm_dir / f"{candidate_id}_response.json"
        response_path.write_text(json.dumps(response.raw_message, indent=2, sort_keys=True), encoding="utf-8")
        search_replace_blocks = parse_search_replace_blocks(response.content)
        edit_format = "unified_diff"
        match_status = "no_search_replace_blocks"
        used_whitespace_tolerance = False
        matched_paths: tuple[str, ...] = ()
        whitespace_tolerant_paths: tuple[str, ...] = ()
        failed_paths: tuple[str, ...] = ()
        if search_replace_blocks:
            search_replace_result = build_search_replace_patch(task_dir, search_replace_blocks)
            edit_format = "search_replace"
            match_status = search_replace_result.match_status
            used_whitespace_tolerance = search_replace_result.used_whitespace_tolerance
            matched_paths = search_replace_result.matched_paths
            whitespace_tolerant_paths = search_replace_result.whitespace_tolerant_paths
            failed_paths = search_replace_result.failed_paths
            patch_text = search_replace_result.patch_text if search_replace_result.all_blocks_matched else ""
        else:
            patch_text = extract_unified_diff(response.content)
        if search_replace_blocks and not patch_text:
            patch_text = response.content
        patch_file = llm_dir / f"{candidate_id}.patch"
        patch_file.write_text(patch_text or response.content, encoding="utf-8")
        trace.emit(
            "llm_candidate_received",
            candidate_id=candidate_id,
            patch_file=str(patch_file),
            extracted_unified_diff=bool(patch_text) and edit_format == "unified_diff",
            edit_format=edit_format,
            search_replace_match_status=match_status,
            search_replace_all_matched=search_replace_result.all_blocks_matched if search_replace_blocks else None,
            search_replace_used_whitespace_tolerance=used_whitespace_tolerance,
            search_replace_matched_paths=matched_paths,
            search_replace_whitespace_tolerant_paths=whitespace_tolerant_paths,
            search_replace_failed_paths=failed_paths,
            response_path=str(response_path),
        )

        result = run_task(task_dir, run_dir / "candidates" / candidate_id, patch_file=patch_file)
        evidence = build_evidence_packet(result.gates)
        failure_type = classify_failure(result.gates)
        fingerprint = _failure_fingerprint(result.gates)
        call_usage = _usage_tokens(response.usage)
        if call_usage is None:
            usage_reported = False
            prompt_tokens = 0
            completion_tokens = _estimate_text_tokens(response.content)
            token_source = "estimate"
        else:
            prompt_tokens, completion_tokens = call_usage
            usage_prompt_tokens += prompt_tokens
            usage_completion_tokens += completion_tokens
            token_source = "provider_usage"
        declared_metadata = parse_generator_metadata(response.content)
        candidate_intent = ""
        candidate_routes: tuple[str, ...] = ()
        metadata_source = "controller"
        if declared_metadata is not None:
            valid_routes = tuple(
                route_name
                for route_name in declared_metadata.compatible_routes
                if route_name in KNOWN_ROUTES
            )
            rejected_routes = tuple(
                route_name
                for route_name in declared_metadata.compatible_routes
                if route_name not in KNOWN_ROUTES
            )
            if rejected_routes:
                trace.emit(
                    "llm_metadata_routes_rejected",
                    candidate_id=candidate_id,
                    rejected_routes=list(rejected_routes),
                )
            declared_intent = declared_metadata.intent or ""
            # Only treat metadata as generator-derived when something usable
            # survives validation; otherwise fall back to controller metadata.
            if declared_intent or valid_routes:
                metadata_source = "generator"
                candidate_intent = declared_intent
                candidate_routes = valid_routes
        record = CandidateRecord(
            candidate_id=candidate_id,
            patch_file=str(patch_file),
            status=result.status,
            route=route,
            failure_type=failure_type,
            score=_score_result(result.status, failure_type),
            gates=tuple({"name": gate.name, "passed": gate.passed} for gate in result.gates),
            parent_id=archive.best_parent_id(),
            generation=archive.next_generation(),
            touched_files=result.touched_files,
            candidate_intent=candidate_intent,
            compatible_routes=candidate_routes,
            metadata_source=metadata_source,
            rationale=_rationale(route, evidence),
            verifier_results=tuple(_verifier_result(gate) for gate in result.gates),
            failure_fingerprint=fingerprint,
            token_estimate=completion_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            token_source=token_source,
            duration_seconds=result.duration_seconds,
            evidence=evidence,
        )
        writer_tokens += _estimate_text_tokens(response.content)
        archive.add(record)
        trace.emit(
            "llm_candidate_evaluated",
            candidate_id=candidate_id,
            status=record.status,
            failure_type=failure_type,
            score=record.score,
            evidence_summary=evidence.summary if evidence else None,
            edit_format=edit_format,
            search_replace_match_status=match_status,
        )

        if result.passed:
            stop_reason = "promoted"
            break
        if archive.repeated_failures(fingerprint or failure_type) >= budget.max_repeated_failures:
            stop_reason = "repeated_failure_limit"
            trace.emit("llm_search_stopped", reason=stop_reason, failure_type=failure_type, fingerprint=fingerprint)
            break

        messages.append(_assistant_message(response, client.reasoning_enabled))
        repair_prompt = _repair_prompt(task_dir, route, response.content, evidence, archive.records)
        writer_tokens += _estimate_text_tokens(repair_prompt)
        messages.append({"role": "user", "content": repair_prompt})
    else:
        stop_reason = "candidate_budget_exhausted"

    has_provider_usage = usage_reported and archive.records and (usage_prompt_tokens or usage_completion_tokens)
    if has_provider_usage:
        prompt_tokens_total = usage_prompt_tokens
        completion_tokens_total = usage_completion_tokens
        total_tokens = usage_prompt_tokens + usage_completion_tokens
        token_source = "provider_usage"
    else:
        prompt_tokens_total = 0
        completion_tokens_total = sum(record.completion_tokens for record in archive.records)
        total_tokens = writer_tokens
        token_source = "estimate"
    # Only price runs backed by complete provider-reported usage. Estimate-only
    # runs have prompt_tokens=0 and estimated completion tokens, so any priced
    # figure would exclude prompt cost and look more authoritative than it is.
    cost = (
        estimate_cost(client.model, prompt_tokens_total, completion_tokens_total, resolved_price_schedule)
        if resolved_price_schedule and has_provider_usage
        else None
    )
    currency = (
        (schedule_currency(client.model, resolved_price_schedule) or "USD")
        if cost is not None
        else None
    )
    wall_seconds = time.monotonic() - started
    archive_path = run_dir / "candidate_archive.json"
    archive.save(
        archive_path,
        run=build_run_metadata(
            task,
            archive.records,
            status="promoted" if archive.records and archive.records[-1].status == "promoted" else "stopped",
            stop_reason=stop_reason,
            total_tokens=total_tokens,
            wall_seconds=wall_seconds,
            budget=budget,
            model=client.model,
            provider=provider,
            temperature=client.temperature,
            prompt_tokens=prompt_tokens_total,
            completion_tokens=completion_tokens_total,
            token_source=token_source,
            cost=cost,
            currency=currency,
            seed=client.seed,
            reasoning_enabled=client.reasoning_enabled,
            endpoint=client.url,
        ),
    )
    status = "promoted" if archive.records and archive.records[-1].status == "promoted" else "stopped"
    trace.emit(
        "llm_search_finished",
        task_id=task.task_id,
        status=status,
        stop_reason=stop_reason,
        archive_path=str(archive_path),
        evaluated_candidates=len(archive.records),
        token_source=token_source,
        total_tokens=total_tokens,
    )
    return SearchResult(
        task_id=task.task_id,
        status=status,
        stop_reason=stop_reason,
        trace_path=trace_path,
        archive_path=archive_path,
        records=archive.records,
        total_tokens=total_tokens,
        prompt_tokens=prompt_tokens_total,
        completion_tokens=completion_tokens_total,
        token_source=token_source,
        cost=cost,
        currency=currency,
        wall_seconds=wall_seconds,
    )


def extract_unified_diff(content: str) -> str:
    fenced = _extract_fenced_diff(content)
    if fenced:
        return _normalize_unified_diff(fenced)
    start = content.find("diff --git ")
    if start == -1:
        start = content.find("--- ")
    if start == -1:
        return ""
    diff = content[start:].strip()
    fence_end = diff.find("\n```")
    if fence_end != -1:
        diff = diff[:fence_end].strip()
    return _normalize_unified_diff(diff)


def parse_search_replace_blocks(content: str) -> list[SearchReplaceBlock]:
    lines = content.splitlines()
    blocks: list[SearchReplaceBlock] = []
    index = 0
    while index < len(lines):
        if lines[index].strip() != "<<<<<<< SEARCH":
            index += 1
            continue
        if index == 0:
            index += 1
            continue
        path_index = index - 1
        while path_index >= 0:
            candidate_path = lines[path_index].strip()
            if candidate_path and not candidate_path.startswith("```"):
                break
            path_index -= 1
        if path_index < 0:
            index += 1
            continue
        path = lines[path_index].strip()
        if path in {"<<<<<<< SEARCH", "=======", ">>>>>>> REPLACE"}:
            index += 1
            continue
        search_lines: list[str] = []
        cursor = index + 1
        while cursor < len(lines) and lines[cursor].strip() != "=======":
            search_lines.append(lines[cursor])
            cursor += 1
        if cursor >= len(lines):
            break
        replace_lines: list[str] = []
        cursor += 1
        while cursor < len(lines) and lines[cursor].strip() != ">>>>>>> REPLACE":
            replace_lines.append(lines[cursor])
            cursor += 1
        if cursor >= len(lines):
            break
        blocks.append(
            SearchReplaceBlock(
                path=path,
                search_text="\n".join(search_lines),
                replace_text="\n".join(replace_lines),
            )
        )
        index = cursor + 1
    return blocks


def build_search_replace_patch(task_dir: Path, blocks: list[SearchReplaceBlock]) -> SearchReplacePatchResult:
    workspace = task_dir / "repo"
    grouped_blocks: dict[str, list[SearchReplaceBlock]] = {}
    ordered_paths: list[str] = []
    for block in blocks:
        if block.path not in grouped_blocks:
            grouped_blocks[block.path] = []
            ordered_paths.append(block.path)
        grouped_blocks[block.path].append(block)

    patch_chunks: list[str] = []
    matched_paths: list[str] = []
    whitespace_tolerant_paths: list[str] = []
    failed_paths: list[str] = []
    for path in ordered_paths:
        file_path = workspace / path
        try:
            original_text = file_path.read_text(encoding="utf-8")
        except OSError:
            # The model named a path that does not exist in the repo (e.g. it
            # emitted prose that was mistaken for a file-path line). Treat this
            # as an unmatched block so the candidate fails its patch gate
            # gracefully instead of crashing the whole search.
            failed_paths.append(path)
            continue
        current_text = original_text
        file_matched = True
        for block in grouped_blocks[path]:
            match = _locate_search_block(current_text, block.search_text)
            if match is None:
                failed_paths.append(path)
                file_matched = False
                break
            start, end, used_whitespace_tolerance = match
            matched_paths.append(path)
            if used_whitespace_tolerance:
                whitespace_tolerant_paths.append(path)
            current_text = current_text[:start] + block.replace_text + current_text[end:]
        if file_matched and current_text != original_text:
            patch_chunks.append(_unified_diff_for_file(path, original_text, current_text))

    all_blocks_matched = not failed_paths
    match_status = "matched" if all_blocks_matched else "search_block_no_match"
    patch_text = "\n".join(chunk for chunk in patch_chunks if chunk) if all_blocks_matched else ""
    return SearchReplacePatchResult(
        patch_text=patch_text,
        all_blocks_matched=all_blocks_matched,
        match_status=match_status,
        used_whitespace_tolerance=bool(whitespace_tolerant_paths),
        matched_paths=tuple(matched_paths),
        whitespace_tolerant_paths=tuple(whitespace_tolerant_paths),
        failed_paths=tuple(failed_paths),
    )


def _locate_search_block(content: str, search_text: str) -> tuple[int, int, bool] | None:
    if not search_text:
        return None
    start = content.find(search_text)
    if start != -1:
        return start, start + len(search_text), False

    content_lines = content.splitlines(keepends=True)
    search_lines = search_text.splitlines(keepends=True)
    if not search_lines:
        return None
    normalized_search = [_normalize_search_line(line) for line in search_lines]
    line_offsets: list[int] = []
    offset = 0
    for line in content_lines:
        line_offsets.append(offset)
        offset += len(line)

    window_size = len(search_lines)
    for index in range(0, len(content_lines) - window_size + 1):
        window = content_lines[index:index + window_size]
        if [_normalize_search_line(line) for line in window] != normalized_search:
            continue
        start = line_offsets[index]
        end = start + sum(len(line) for line in window)
        return start, end, True
    return None


def _normalize_search_line(line: str) -> str:
    return line.rstrip().strip()


def _unified_diff_for_file(path: str, original_text: str, modified_text: str) -> str:
    original_lines = original_text.splitlines()
    modified_lines = modified_text.splitlines()
    diff_lines = [f"diff --git a/{path} b/{path}"]
    diff_lines.extend(
        line.rstrip("\n")
        for line in difflib.unified_diff(
            original_lines,
            modified_lines,
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            lineterm="\n",
        )
    )
    return "\n".join(diff_lines) + "\n"


def _extract_fenced_diff(content: str) -> str:
    for fence in ("```diff", "```patch", "```"):
        start = content.find(fence)
        if start == -1:
            continue
        body_start = content.find("\n", start)
        if body_start == -1:
            continue
        end = content.find("```", body_start + 1)
        if end == -1:
            continue
        candidate = content[body_start + 1 : end].strip()
        if candidate.startswith(("diff --git ", "--- ")):
            return f"{candidate}\n"
    return ""


def _normalize_unified_diff(diff: str) -> str:
    lines = diff.strip().splitlines()
    if not lines:
        return ""
    lines = _ensure_git_diff_header(lines)
    return f"{_recount_hunks(lines)}\n"


def _ensure_git_diff_header(lines: list[str]) -> list[str]:
    if lines[0].startswith("diff --git "):
        return lines
    if len(lines) < 2 or not lines[0].startswith("--- ") or not lines[1].startswith("+++ "):
        return lines

    old_path = _diff_path(lines[0][4:].strip())
    new_path = _diff_path(lines[1][4:].strip())
    lines = lines.copy()
    lines[0] = f"--- a/{old_path}" if old_path != "/dev/null" else "--- /dev/null"
    lines[1] = f"+++ b/{new_path}" if new_path != "/dev/null" else "+++ /dev/null"
    return [f"diff --git a/{old_path} b/{new_path}", *lines]


def _diff_path(path: str) -> str:
    if path == "/dev/null":
        return path
    if path.startswith(("a/", "b/")):
        return path[2:]
    return path


def _recount_hunks(lines: list[str]) -> str:
    recounted: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.startswith("@@"):
            recounted.append(line)
            index += 1
            continue

        hunk_lines: list[str] = []
        index += 1
        while index < len(lines) and not lines[index].startswith(("@@", "diff --git ")):
            hunk_lines.append(lines[index])
            index += 1
        old_count = sum(1 for hunk_line in hunk_lines if hunk_line.startswith((" ", "-")))
        new_count = sum(1 for hunk_line in hunk_lines if hunk_line.startswith((" ", "+")))
        recounted.append(_rewrite_hunk_header(line, old_count, new_count))
        recounted.extend(hunk_lines)
    return "\n".join(recounted)


def _rewrite_hunk_header(header: str, old_count: int, new_count: int) -> str:
    parts = header.split("@@")
    ranges = parts[1].strip().split()
    if len(ranges) < 2:
        return f"@@ -1,{old_count} +1,{new_count} @@"
    old_start = ranges[0].split(",", maxsplit=1)[0]
    new_start = ranges[1].split(",", maxsplit=1)[0]
    suffix = f" {parts[2].strip()}" if len(parts) > 2 and parts[2].strip() else ""
    return f"@@ {old_start},{old_count} {new_start},{new_count} @@{suffix}"


def _trim_chat_history(messages: list[dict[str, object]], prompt_token_cap: int) -> None:
    if len(messages) <= 3:
        # Trim the single (initial) user prompt from the end to keep instructions.
        prompt_tokens = sum(_estimate_text_tokens(str(message.get("content", ""))) for message in messages)
        while prompt_tokens > prompt_token_cap:
            for message in reversed(messages):
                if message.get("role") == "user" and isinstance(message.get("content"), str):
                    content = str(message["content"])
                    lines = content.splitlines()
                    if len(lines) > 1:
                        lines.pop()
                        content = "\n".join(lines)
                    else:
                        content = content[: max(len(content) // 2, 1)]
                    message["content"] = content
                    break
            prompt_tokens = sum(_estimate_text_tokens(str(message.get("content", ""))) for message in messages)
        return
    while len(messages) > 3:
        prompt_tokens = sum(_estimate_text_tokens(str(message.get("content", ""))) for message in messages)
        if prompt_tokens <= prompt_token_cap:
            return
        del messages[1:3]


def _assistant_message(response: LLMResponse, preserve_reasoning: bool) -> dict[str, object]:
    message: dict[str, object] = {"role": "assistant", "content": response.content}
    if preserve_reasoning and response.reasoning_details is not None:
        message["reasoning_details"] = response.reasoning_details
    return message


def _initial_prompt(task_dir: Path) -> str:
    task = load_task(task_dir)
    workspace = task_dir / "repo"
    packets = context_packet_index(task_dir)
    packet_summary = "\n".join(f"- {packet_id}: {packet['kind']} {packet['path']}" for packet_id, packet in packets.items())
    allowed = ", ".join(task.allowed_paths)
    source_sections = [
        _section(f"repo/{relative}", (workspace / relative).read_text(encoding="utf-8"))
        for relative in task.allowed_paths
        if (workspace / relative).exists()
    ]
    visible_sections = [
        _section(f"repo/{path.relative_to(workspace)}", path.read_text(encoding="utf-8"))
        for test_dir in task.visible_tests
        for path in sorted((workspace / test_dir).glob("test_*.py"))
    ]
    spec_path = workspace / "spec.md"
    spec_sections = (
        [_section("repo/spec.md", spec_path.read_text(encoding="utf-8"))] if spec_path.exists() else []
    )
    return "\n\n".join(
        [
            "You are generating a candidate repair patch for GateGRPO.",
            "Return only Aider-style SEARCH/REPLACE blocks in the exact format below. Do not include prose or unified diffs.",
            "For each block, copy the SEARCH section verbatim from the provided source and make it specific enough to identify the target edit.",
            "Use one block per change.",
            "When the spec requires a specific rounding rule, implement that exact rule; Python's built-in round() uses banker's rounding and int()/float truncation drops the fraction, so prefer exact integer arithmetic such as (numerator + divisor // 2) // divisor to avoid floating-point error.",
            "Implement every rule in the spec, including all error and validation cases: raise exactly the specified exception (e.g. ValueError) for each invalid input the spec describes, even when no provided visible test exercises that case.",
            "When splitting delimited text whose rows are wrapped in a leading and trailing delimiter (for example a table row like |a|b|), the split yields an empty fragment before the first delimiter and after the last one; strip only that outer boundary delimiter before splitting (or drop exactly those two boundary fragments), and never filter out empty fields or use truthiness to skip fields, so that legitimate interior and trailing empty cells are preserved and the cell count matches the header.",
            "When a delimiter can be escaped (for example \\| meaning a literal pipe inside a cell), split on unescaped delimiters only; do not replace the escape sequence with the literal delimiter before splitting, or that now-literal character will be treated as a separator and create an extra cell (which also breaks any cell-count comparison). Split first while respecting the escape, then unescape the sequence to its literal character within each resulting field.",
            "Format:",
            "relative/path/from/repo/root",
            "<<<<<<< SEARCH",
            "exact original lines",
            "=======",
            "replacement lines",
            ">>>>>>> REPLACE",
            _GENERATOR_METADATA_INSTRUCTION,
            f"The patch must modify only these allowed paths: {allowed}.",
            _section("evidence packet index", packet_summary),
            _section("instructions.md", (task_dir / "instructions.md").read_text(encoding="utf-8")),
            *spec_sections,
            *source_sections,
            *visible_sections,
            "Release-gate regression tests exist, but their source and assertion details are withheld.",
        ]
    )


def _repair_prompt(
    task_dir: Path,
    route: str,
    previous_patch: str,
    evidence: EvidencePacket | None,
    prior_attempts: list[CandidateRecord] | None = None,
) -> str:
    evidence_summary = "No structured evidence was available."
    if evidence is not None:
        evidence_summary = json.dumps(
            {
                "gate": evidence.gate,
                "failure_type": evidence.failure_type,
                "summary": evidence.summary,
                "details": evidence.details,
            },
            indent=2,
            sort_keys=True,
        )
    task = load_task(task_dir)
    workspace = task_dir / "repo"
    allowed = ", ".join(task.allowed_paths)
    source_sections = [
        _section(f"repo/{relative}", (workspace / relative).read_text(encoding="utf-8"))
        for relative in task.allowed_paths
        if (workspace / relative).exists()
    ]
    visible_sections = [
        _section(f"repo/{path.relative_to(workspace)}", path.read_text(encoding="utf-8"))
        for test_dir in task.visible_tests
        for path in sorted((workspace / test_dir).glob("test_*.py"))
    ]
    attempted_sections = _prior_attempts_section(prior_attempts or [])
    return "\n\n".join(
        [
            f"The previous patch was rejected. Route: {route}.",
            "You may reason briefly before the final answer, but the response must end with the exact SEARCH/REPLACE block(s) below.",
            "Do not include unified diffs.",
            "Do not place any prose between a file path line and its <<<<<<< SEARCH marker; the file path must appear immediately before the marker.",
            "Copy the SEARCH section verbatim from the provided source and make each block specific enough to identify the target edit.",
            "Each attempt is applied independently to the ORIGINAL source files shown below. SEARCH must match the original source verbatim, not the previous attempt's edited version.",
            "Do not resubmit any approach listed as already attempted and rejected below.",
            "If multiple attempts failed the same gate or test, the core algorithm or data structure is likely wrong; change the approach rather than making a small tweak.",
            "Reconsider the overall approach instead of making a small tweak to the same idea.",
            "Before finalizing, mentally execute the edited code on each provided visible test input and confirm it produces exactly the expected output shown there; if any case differs, revise the edit before answering.",
            "Read the failing test's expected-versus-actual output in the failure evidence carefully.",
            "When the spec requires a specific rounding rule, implement that exact rule; Python's built-in round() uses banker's rounding and int()/float truncation drops the fraction, so prefer exact integer arithmetic such as (numerator + divisor // 2) // divisor to avoid floating-point error.",
            "If visible tests passed but a withheld release-gate regression failed, use the failing test identifiers in the evidence to pinpoint the regressed spec behavior and fix that while preserving passing behaviors; implement every rule in the spec, including all error and validation cases (raise exactly the specified exception for each invalid input the spec describes), not only the behavior the visible tests exercise.",
            "Re-check every rule in the spec before writing the next attempt.",
            "When splitting delimited text whose rows are wrapped in a leading and trailing delimiter (for example a table row like |a|b|), the split yields an empty fragment before the first delimiter and after the last one; strip only that outer boundary delimiter before splitting (or drop exactly those two boundary fragments), and never filter out empty fields or use truthiness to skip fields, so that legitimate interior and trailing empty cells are preserved and the cell count matches the header.",
            "When a delimiter can be escaped (for example \\| meaning a literal pipe inside a cell), split on unescaped delimiters only; do not replace the escape sequence with the literal delimiter before splitting, or that now-literal character will be treated as a separator and create an extra cell (which also breaks any cell-count comparison). Split first while respecting the escape, then unescape the sequence to its literal character within each resulting field.",
            "Format:",
            "relative/path/from/repo/root",
            "<<<<<<< SEARCH",
            "exact original lines",
            "=======",
            "replacement lines",
            ">>>>>>> REPLACE",
            _GENERATOR_METADATA_INSTRUCTION,
            f"The patch must modify only these allowed paths: {allowed}.",
            *source_sections,
            *visible_sections,
            attempted_sections,
            _section("failure evidence", evidence_summary),
            "Modify only the task's allowed source file(s).",
        ]
    )


def _prior_attempts_section(prior_attempts: list[CandidateRecord]) -> str:
    if not prior_attempts:
        return _section("Approaches already attempted and rejected — do NOT repeat these", "No prior failed attempts were recorded.")
    entries: list[str] = []
    for attempt_number, record in enumerate(prior_attempts, start=1):
        patch_text = ""
        try:
            patch_text = Path(record.patch_file).read_text(encoding="utf-8")
        except OSError:
            patch_text = f"<unable to read patch file: {record.patch_file}>"
        entries.append(
            "\n".join(
                [
                    f"Attempt {attempt_number}: route={record.route}, failure_type={record.failure_type or 'unknown'}",
                    "Patch snippet:",
                    _truncate_text(patch_text, 600),
                ]
            )
        )
    return _section("Approaches already attempted and rejected — do NOT repeat these", "\n\n".join(entries))


def _truncate_text(content: str, limit: int) -> str:
    if len(content) <= limit:
        return content
    return content[:limit].rstrip() + "\n...[truncated]"


def _section(name: str, content: str) -> str:
    return f"## {name}\n```text\n{content.strip()}\n```"


def parse_generator_metadata(content: str) -> GeneratorMetadata | None:
    """Parse an optional generator-emitted metadata block from a response.

    The model may declare metadata about its own candidate using a fenced
    block, for example::

        ```gategrpo-metadata
        {"intent": "complete_spec_repair", "compatible_routes": ["behavior_repair"]}
        ```

    When present, this metadata is generator-derived rather than
    controller-derived. A missing or malformed block yields ``None`` so the
    controller falls back to its own derivation.
    """
    marker = "```gategrpo-metadata"
    start = content.find(marker)
    if start == -1:
        return None
    body_start = content.find("\n", start)
    if body_start == -1:
        return None
    end = content.find("```", body_start + 1)
    if end == -1:
        return None
    block = content[body_start + 1 : end].strip()
    try:
        parsed = json.loads(block)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    intent = parsed.get("intent")
    routes = parsed.get("compatible_routes")
    intent_value = intent if isinstance(intent, str) and intent.strip() else None
    routes_value: tuple[str, ...] = ()
    if isinstance(routes, list):
        routes_value = tuple(str(route) for route in routes if isinstance(route, str) and route.strip())
    if intent_value is None and not routes_value:
        return None
    return GeneratorMetadata(intent=intent_value, compatible_routes=routes_value)


def _usage_tokens(usage: dict[str, object] | None) -> tuple[int, int] | None:
    """Return (prompt_tokens, completion_tokens) from a provider usage payload."""
    if not isinstance(usage, dict):
        return None
    prompt = _coerce_token_count(usage.get("prompt_tokens"))
    completion = _coerce_token_count(usage.get("completion_tokens"))
    if prompt is None and completion is None:
        total = _coerce_token_count(usage.get("total_tokens"))
        if total is None:
            return None
        return 0, total
    return prompt or 0, completion or 0


def _coerce_token_count(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _extract_usage(raw: object) -> dict[str, object] | None:
    if not isinstance(raw, dict):
        return None
    usage = raw.get("usage")
    if isinstance(usage, dict):
        return {str(key): value for key, value in usage.items()}
    return None


def _extract_message(raw: object) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise RuntimeError("OpenRouter response was not a JSON object")
    choices = raw.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"OpenRouter response did not include choices: {_summarize_openrouter_payload(raw)}")
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise RuntimeError("OpenRouter response choice was malformed")
    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise RuntimeError("OpenRouter response did not include a message")
    return {str(key): value for key, value in message.items()}


def _read_error_json(exc: urllib.error.HTTPError) -> object:
    body = exc.read().decode("utf-8", errors="replace")
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return {"error": {"message": body[-1000:]}}


def _summarize_openrouter_payload(raw: object) -> str:
    if not isinstance(raw, dict):
        return "unexpected non-object response"
    error = raw.get("error")
    if isinstance(error, dict):
        parts = []
        for key in ("code", "message", "type"):
            value = error.get(key)
            if isinstance(value, (str, int, float)):
                parts.append(f"{key}={value}")
        metadata = error.get("metadata")
        if isinstance(metadata, dict):
            provider = metadata.get("provider_name") or metadata.get("provider")
            if isinstance(provider, str):
                parts.append(f"provider={provider}")
        return "; ".join(parts) if parts else "error object present"
    message = raw.get("message")
    if isinstance(message, str):
        return message
    return f"keys={sorted(str(key) for key in raw.keys())}"


def _score_result(status: str, failure_type: str | None) -> float:
    if status == "promoted":
        return 1.0
    if failure_type == "release_gate_regressions":
        return 0.6
    if failure_type == "visible_tests":
        return 0.4
    if failure_type in {"python_ast", "secret_scan", "scope_guard"}:
        return 0.1
    return 0.0


def _failure_fingerprint(gates) -> str | None:
    for gate in gates:
        if not gate.passed:
            return gate.fingerprint or gate.name
    return None


def _rationale(route: str, evidence: EvidencePacket | None) -> str:
    if evidence is None:
        return "candidate passed all hard gates"
    return f"{route}: respond to {evidence.summary}"


def _verifier_result(gate) -> dict[str, object]:
    return {"name": gate.name, "passed": gate.passed, "fingerprint": gate.fingerprint}


def _estimate_text_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _wall_budget_exhausted(started: float, budget: SearchBudget) -> bool:
    if budget.max_wall_seconds is None:
        return False
    return (time.monotonic() - started) >= budget.max_wall_seconds
