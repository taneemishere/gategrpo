import json
import subprocess
from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace

import pytest

from gategrpo.llm import (
    LLMResponse,
    CachedLLMClient,
    GeneratorMetadata,
    OPENROUTER_MODEL,
    VLLM_DEFAULT_BASE_URL,
    VLLM_DEFAULT_MODEL,
    OpenAICompatibleClient,
    OpenRouterClient,
    build_llm_client,
    build_search_replace_patch,
    extract_unified_diff,
    parse_generator_metadata,
    parse_search_replace_blocks,
    _repair_prompt,
    _usage_tokens,
    run_llm_search,
    SearchReplaceBlock,
)


ROOT = Path(__file__).resolve().parents[1]
SMOKE_TASK = ROOT / "tasks" / "smoke_markdown_parser"


class FakeClient:
    model = "test-model"
    reasoning_enabled = True
    temperature = None
    seed = None
    url = "fake://test"

    def __init__(self, patch_text: str) -> None:
        self.patch_text = patch_text
        self.messages_seen: list[list[dict]] = []

    def chat(self, messages):
        self.messages_seen.append(messages)
        return LLMResponse(
            content=f"```diff\n{self.patch_text}\n```",
            reasoning_details=[{"type": "summary", "text": "generated patch"}],
            raw_message={"content": self.patch_text, "reasoning_details": [{"type": "summary", "text": "generated patch"}]},
        )


def test_extract_unified_diff_from_fenced_response():
    content = "Here is the patch:\n```diff\ndiff --git a/parser.py b/parser.py\n--- a/parser.py\n+++ b/parser.py\n@@ -1 +1 @@\n-old\n+new\n```\n"

    patch = extract_unified_diff(content)

    assert patch.startswith("diff --git a/parser.py b/parser.py")
    assert "Here is the patch" not in patch


def test_extract_unified_diff_normalizes_plain_unified_diff():
    content = """--- parser.py
+++ parser.py
@@ -1,99 +1,99 @@
-old
+new
"""

    patch = extract_unified_diff(content)

    assert patch.splitlines()[0] == "diff --git a/parser.py b/parser.py"
    assert "--- a/parser.py" in patch
    assert "+++ b/parser.py" in patch
    assert "@@ -1,1 +1,1 @@" in patch


def test_extract_unified_diff_normalizes_bare_hunk_header():
    content = """--- a/parser.py
+++ b/parser.py
@@
-old
+new
"""

    patch = extract_unified_diff(content)

    assert "@@ -1,1 +1,1 @@" in patch


def test_parse_search_replace_blocks_supports_multiple_fenced_and_unfenced_blocks():
    content = dedent(
        """
        ```text
        parser.py
        <<<<<<< SEARCH
        old parser
        =======
        new parser
        >>>>>>> REPLACE
        ```
        utils.py
        <<<<<<< SEARCH
        old utils
        =======
        new utils
        >>>>>>> REPLACE
        """
    ).strip()

    blocks = parse_search_replace_blocks(content)

    assert blocks == [
        SearchReplaceBlock(path="parser.py", search_text="old parser", replace_text="new parser"),
        SearchReplaceBlock(path="utils.py", search_text="old utils", replace_text="new utils"),
    ]


def test_parse_search_replace_blocks_supports_blank_line_before_marker():
    content = dedent(
        """
        ranges.py

        <<<<<<< SEARCH
        old ranges
        =======
        new ranges
        >>>>>>> REPLACE
        """
    ).strip()

    blocks = parse_search_replace_blocks(content)

    assert blocks == [
        SearchReplaceBlock(path="ranges.py", search_text="old ranges", replace_text="new ranges"),
    ]


def test_parse_search_replace_blocks_supports_fenced_block_with_blank_separator():
    content = dedent(
        """
        ```python

        ranges.py

        <<<<<<< SEARCH
        old ranges
        =======
        new ranges
        >>>>>>> REPLACE
        ```
        """
    ).strip()

    blocks = parse_search_replace_blocks(content)

    assert blocks == [
        SearchReplaceBlock(path="ranges.py", search_text="old ranges", replace_text="new ranges"),
    ]


def test_build_search_replace_patch_applies_with_git_apply(tmp_path):
    original = (SMOKE_TASK / "repo" / "parser.py").read_text(encoding="utf-8")
    replacement_text = dedent(
        r'''
        def _split_row(line):
            cells = []
            current = []
            escaped = False
            inner = line.strip()
            if inner.startswith("|"):
                inner = inner[1:]
            if inner.endswith("|"):
                inner = inner[:-1]

            for char in inner:
                if escaped:
                    current.append(char)
                    escaped = False
                    continue
                if char == "\\":
                    escaped = True
                    continue
                if char == "|":
                    cells.append("".join(current).strip())
                    current = []
                    continue
                current.append(char)

            if escaped:
                current.append("\\")
            cells.append("".join(current).strip())
            return cells


        def parse_table(markdown):
            """Parse a simple Markdown table into row dictionaries."""
            table_lines = [line.strip() for line in markdown.splitlines() if line.strip().startswith("|")]
        '''
    ).strip()
    task_dir = tmp_path / "task"
    repo_dir = task_dir / "repo"
    repo_dir.mkdir(parents=True)
    (repo_dir / "parser.py").write_text(original, encoding="utf-8")

    repo = tmp_path / "gitrepo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)
    (repo / "parser.py").write_text(original, encoding="utf-8")

    blocks = [
        SearchReplaceBlock(
            path="parser.py",
            search_text=dedent(
                '''
                def parse_table(markdown):
                    """Parse a simple Markdown table into row dictionaries."""
                    lines = [line.strip() for line in markdown.splitlines() if line.strip()]
                    table_lines = [line for line in lines if line.startswith("|")]
                '''
            ).strip(),
            replace_text=replacement_text,
        )
    ]
    result = build_search_replace_patch(task_dir, blocks)
    patch_path = tmp_path / "candidate.patch"
    patch_path.write_text(result.patch_text, encoding="utf-8")

    subprocess.run(["git", "apply", "--check", str(patch_path)], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(["git", "apply", str(patch_path)], cwd=repo, check=True, capture_output=True, text=True)
    expected = "\n".join(replacement_text.splitlines() + original.splitlines()[4:]) + "\n"
    assert (repo / "parser.py").read_text(encoding="utf-8") == expected


def test_build_search_replace_patch_uses_whitespace_tolerant_matching(tmp_path):
    task_dir = tmp_path / "task"
    repo_dir = task_dir / "repo"
    repo_dir.mkdir(parents=True)
    (repo_dir / "parser.py").write_text("def f():\n    return 1\n", encoding="utf-8")

    blocks = [
        SearchReplaceBlock(
            path="parser.py",
            search_text="def f():  \n\treturn 1   ",
            replace_text="def f():\n    return 2",
        )
    ]

    result = build_search_replace_patch(task_dir, blocks)

    assert result.all_blocks_matched is True
    assert result.used_whitespace_tolerance is True
    assert result.whitespace_tolerant_paths == ("parser.py",)
    assert "diff --git a/parser.py b/parser.py" in result.patch_text


def test_build_search_replace_patch_applies_with_blank_line_path_resolution(tmp_path):
    task_dir = ROOT / "tasks" / "range_parser"
    original = (task_dir / "repo" / "ranges.py").read_text(encoding="utf-8")
    repo = tmp_path / "gitrepo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)
    (repo / "ranges.py").write_text(original, encoding="utf-8")

    search_text = dedent(
        """
        def parse_ranges(text):
            \"\"\"Parse comma-separated integers into a list.\"\"\"
            values = []
            for part in text.split(\",\"):
                values.append(int(part))
            return values
        """
    ).strip()
    replace_text = dedent(
        """
        def parse_ranges(text):
            \"\"\"Parse comma-separated integers into a list.\"\"\"
            values = []
            for part in text.split(\",\"):
                part = part.strip()
                if not part:
                    continue
                values.append(int(part))
            return values
        """
    ).strip()
    blocks = parse_search_replace_blocks(
        "ranges.py\n\n<<<<<<< SEARCH\n"
        + search_text
        + "\n=======\n"
        + replace_text
        + "\n>>>>>>> REPLACE"
    )

    result = build_search_replace_patch(task_dir, blocks)
    patch_path = tmp_path / "candidate.patch"
    patch_path.write_text(result.patch_text, encoding="utf-8")

    subprocess.run(["git", "apply", "--check", str(patch_path)], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(["git", "apply", str(patch_path)], cwd=repo, check=True, capture_output=True, text=True)
    assert result.all_blocks_matched is True
    assert result.match_status == "matched"
    assert (repo / "ranges.py").read_text(encoding="utf-8") == original.replace(search_text, replace_text)


def test_build_search_replace_patch_reports_search_miss(tmp_path):
    task_dir = tmp_path / "task"
    repo_dir = task_dir / "repo"
    repo_dir.mkdir(parents=True)
    (repo_dir / "parser.py").write_text("def f():\n    return 1\n", encoding="utf-8")

    blocks = [
        SearchReplaceBlock(
            path="parser.py",
            search_text="def missing():\n    return 1",
            replace_text="def missing():\n    return 2",
        )
    ]

    result = build_search_replace_patch(task_dir, blocks)

    assert result.all_blocks_matched is False
    assert result.match_status == "search_block_no_match"
    assert result.patch_text == ""
    assert result.failed_paths == ("parser.py",)


def test_build_search_replace_patch_reports_missing_path_without_raising(tmp_path):
    task_dir = SMOKE_TASK
    blocks = [
        SearchReplaceBlock(
            path="does_not_exist.py",
            search_text="old text",
            replace_text="new text",
        )
    ]

    result = build_search_replace_patch(task_dir, blocks)

    assert result.all_blocks_matched is False
    assert result.patch_text == ""
    assert result.failed_paths == ("does_not_exist.py",)


def test_repair_prompt_reincludes_original_source_and_anchoring_instruction():
    prompt = _repair_prompt(SMOKE_TASK, "initial_repair", "previous patch", None, [])

    assert "You may reason briefly before the final answer, but the response must end with the exact SEARCH/REPLACE block(s) below." in prompt
    assert "Do not place any prose between a file path line and its <<<<<<< SEARCH marker" in prompt
    assert "Before finalizing, mentally execute the edited code on each provided visible test input and confirm it produces exactly the expected output shown there" in prompt
    assert "Each attempt is applied independently to the ORIGINAL source files shown below" in prompt
    assert "SEARCH must match the original source verbatim" in prompt
    assert "Do not resubmit any approach listed as already attempted and rejected below." in prompt
    assert "If multiple attempts failed the same gate or test, the core algorithm or data structure is likely wrong" in prompt
    assert "Reconsider the overall approach instead of making a small tweak to the same idea." in prompt
    assert "Read the failing test's expected-versus-actual output in the failure evidence carefully." in prompt
    assert "Re-check every rule in the spec before writing the next attempt." in prompt
    assert "repo/parser.py" in prompt
    assert "def parse_table(markdown):" in prompt
    assert "test_parses_basic_markdown_table" in prompt


def test_repair_prompt_includes_prior_attempt_history(tmp_path):
    patch_path = tmp_path / "attempt.patch"
    patch_path.write_text("diff --git a/parser.py b/parser.py\n-old\n+new\n", encoding="utf-8")
    prior_attempt = SimpleNamespace(
        patch_file=str(patch_path),
        route="behavior_repair",
        failure_type="visible_tests",
    )

    prompt = _repair_prompt(SMOKE_TASK, "behavior_repair", "previous patch", None, [prior_attempt])

    assert "Approaches already attempted and rejected — do NOT repeat these" in prompt
    assert "Attempt 1: route=behavior_repair, failure_type=visible_tests" in prompt
    assert "diff --git a/parser.py b/parser.py" in prompt
    assert "-old" in prompt
    assert "core algorithm or data structure is likely wrong" in prompt
    assert "You may reason briefly before the final answer, but the response must end with the exact SEARCH/REPLACE block(s) below." in prompt

def test_run_llm_search_promotes_generated_patch(tmp_path):
    client = FakeClient((SMOKE_TASK / "candidate.patch").read_text(encoding="utf-8"))

    result = run_llm_search(SMOKE_TASK, tmp_path / "llm_search", client=client)

    assert result.status == "promoted"
    assert result.records[0].candidate_id == "llm_candidate_001"
    assert result.records[0].route == "initial_repair"
    assert (tmp_path / "llm_search" / "llm" / "llm_candidate_001.patch").exists()
    assert client.messages_seen[0][0]["role"] == "user"


def test_run_llm_search_promotes_search_replace_patch(tmp_path):
    original = (SMOKE_TASK / "repo" / "parser.py").read_text(encoding="utf-8")
    golden_repo = tmp_path / "golden_repo"
    golden_repo.mkdir()
    subprocess.run(["git", "init"], cwd=golden_repo, check=True, capture_output=True, text=True)
    (golden_repo / "parser.py").write_text(original, encoding="utf-8")
    subprocess.run(["git", "apply", str(SMOKE_TASK / "candidate.patch")], cwd=golden_repo, check=True, capture_output=True, text=True)
    replacement_text = (golden_repo / "parser.py").read_text(encoding="utf-8")
    content = (
        "```text\n"
        "parser.py\n"
        "<<<<<<< SEARCH\n"
        f"{original}\n"
        "=======\n"
        f"{replacement_text}"
        ">>>>>>> REPLACE\n"
        "```"
    )

    class FakeSearchReplaceClient:
        model = "test-model"
        reasoning_enabled = True
        temperature = None
        seed = None
        url = "fake://test"

        def __init__(self, response_text: str) -> None:
            self.response_text = response_text
            self.messages_seen: list[list[dict]] = []

        def chat(self, messages):
            self.messages_seen.append(messages)
            return LLMResponse(
                content=self.response_text,
                reasoning_details=[{"type": "summary", "text": "generated patch"}],
                raw_message={"content": self.response_text},
            )

    client = FakeSearchReplaceClient(content)
    result = run_llm_search(SMOKE_TASK, tmp_path / "llm_search_sr", client=client)

    assert result.status == "promoted"
    assert result.records[0].candidate_id == "llm_candidate_001"
    assert result.records[0].route == "initial_repair"
    assert (tmp_path / "llm_search_sr" / "llm" / "llm_candidate_001.patch").exists()
    assert client.messages_seen[0][0]["role"] == "user"


class FakeHTTPResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_openai_compatible_client_posts_expected_payload_and_headers():
    seen_requests = []

    def fake_opener(request, timeout):
        seen_requests.append(request)
        return FakeHTTPResponse({"choices": [{"message": {"content": "diff --git a/parser.py b/parser.py"}}]})

    client = OpenAICompatibleClient(
        base_url="http://example.com/v1/",
        model="demo-model",
        api_key=None,
        reasoning_enabled=True,
        opener=fake_opener,
    )
    response = client.chat([{"role": "user", "content": "patch please"}])

    request = seen_requests[0]
    assert request.full_url == "http://example.com/v1/chat/completions"
    assert request.headers.get("Authorization") is None
    assert "Authorization" not in request.headers
    assert json.loads(request.data.decode("utf-8")) == {
        "model": "demo-model",
        "messages": [{"role": "user", "content": "patch please"}],
    }
    assert response.content.startswith("diff --git")


def test_openai_compatible_client_includes_temperature_when_set():
    seen_requests = []

    def fake_opener(request, timeout):
        seen_requests.append(request)
        return FakeHTTPResponse({"choices": [{"message": {"content": "diff --git a/parser.py b/parser.py"}}]})

    client = OpenAICompatibleClient(
        base_url="http://example.com/v1",
        model="demo-model",
        api_key=None,
        temperature=0.7,
        opener=fake_opener,
    )
    client.chat([{"role": "user", "content": "patch please"}])

    payload = json.loads(seen_requests[0].data.decode("utf-8"))
    assert payload["temperature"] == 0.7
    assert payload["model"] == "demo-model"


def test_openai_compatible_client_adds_bearer_header_when_api_key_present():
    seen_requests = []

    def fake_opener(request, timeout):
        seen_requests.append(request)
        return FakeHTTPResponse({"choices": [{"message": {"content": "diff --git a/parser.py b/parser.py"}}]})

    client = OpenAICompatibleClient(
        base_url="http://example.com/v1",
        model="demo-model",
        api_key="secret-token",
        opener=fake_opener,
    )
    client.chat([{"role": "user", "content": "patch please"}])

    request = seen_requests[0]
    assert request.full_url == "http://example.com/v1/chat/completions"
    assert request.headers.get("Authorization") == "Bearer secret-token"


def test_openrouter_client_can_disable_reasoning():
    seen_payloads = []

    def fake_opener(request, timeout):
        seen_payloads.append(json.loads(request.data.decode("utf-8")))
        return FakeHTTPResponse({"choices": [{"message": {"content": "diff --git a/parser.py b/parser.py"}}]})

    client = OpenRouterClient("test-key", reasoning_enabled=False, opener=fake_opener)
    response = client.chat([{"role": "user", "content": "patch please"}])

    assert client.model == OPENROUTER_MODEL
    assert "reasoning" not in seen_payloads[0]
    assert response.content.startswith("diff --git")


def test_openrouter_client_reports_error_payload_without_choices():
    def fake_opener(request, timeout):
        return FakeHTTPResponse({"error": {"code": 429, "message": "Rate limit exceeded"}})

    client = OpenRouterClient("test-key", opener=fake_opener)

    with pytest.raises(RuntimeError, match="Rate limit exceeded"):
        client.chat([{"role": "user", "content": "patch please"}])


def test_build_llm_client_resolves_provider_defaults(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    vllm_client = build_llm_client(
        "vllm",
        model=None,
        base_url=None,
        api_key=None,
        reasoning_enabled=False,
    )
    assert isinstance(vllm_client, OpenAICompatibleClient)
    assert vllm_client.model == VLLM_DEFAULT_MODEL
    assert vllm_client.base_url == VLLM_DEFAULT_BASE_URL
    assert vllm_client.temperature is None

    openrouter_client = build_llm_client(
        "openrouter",
        model=None,
        base_url=None,
        api_key=None,
        reasoning_enabled=False,
    )
    assert isinstance(openrouter_client, OpenRouterClient)
    assert openrouter_client.model == OPENROUTER_MODEL


def test_build_llm_client_propagates_temperature(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    vllm_client = build_llm_client(
        "vllm",
        model=None,
        base_url=None,
        api_key=None,
        reasoning_enabled=False,
        temperature=0.25,
    )
    assert isinstance(vllm_client, OpenAICompatibleClient)
    assert vllm_client.temperature == 0.25

    openrouter_client = build_llm_client(
        "openrouter",
        model=None,
        base_url=None,
        api_key=None,
        reasoning_enabled=False,
        temperature=0.25,
    )
    assert isinstance(openrouter_client, OpenRouterClient)
    assert openrouter_client.temperature == 0.25


def test_cached_llm_client_uses_disk_cache_for_identical_messages(tmp_path):
    class CountingClient:
        model = "cache-model"
        reasoning_enabled = False

        def __init__(self) -> None:
            self.calls = 0

        def chat(self, messages):
            self.calls += 1
            return LLMResponse(
                content=f"response-{self.calls}",
                reasoning_details={"call": self.calls},
                raw_message={"call": self.calls, "messages": messages},
            )

    inner = CountingClient()
    cache_dir = tmp_path / "cache"
    client = CachedLLMClient(inner, cache_dir)
    messages = [{"role": "user", "content": "hello"}]

    first = client.chat(messages)
    second = client.chat(messages)

    assert inner.calls == 1
    assert first == second
    assert first.content == "response-1"
    assert second.raw_message["call"] == 1

    fresh_inner = CountingClient()
    fresh_client = CachedLLMClient(fresh_inner, cache_dir)
    third = fresh_client.chat(messages)

    assert fresh_inner.calls == 0
    assert third == first
    assert third.content == "response-1"


def test_cached_llm_client_misses_on_different_messages(tmp_path):
    class CountingClient:
        model = "cache-model"
        reasoning_enabled = False

        def __init__(self) -> None:
            self.calls = 0

        def chat(self, messages):
            self.calls += 1
            return LLMResponse(
                content=f"response-{self.calls}",
                reasoning_details=None,
                raw_message={"call": self.calls, "messages": messages},
            )

    client = CachedLLMClient(CountingClient(), tmp_path / "cache")

    first = client.chat([{ "role": "user", "content": "one" }])
    second = client.chat([{ "role": "user", "content": "two" }])

    assert first.content == "response-1"
    assert second.content == "response-2"


def test_build_llm_client_wraps_cached_client_when_requested(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    plain_client = build_llm_client(
        "vllm",
        model=None,
        base_url=None,
        api_key=None,
        reasoning_enabled=False,
    )
    cached_client = build_llm_client(
        "vllm",
        model=None,
        base_url=None,
        api_key=None,
        reasoning_enabled=False,
        cache_dir=tmp_path / "cache",
    )

    assert isinstance(plain_client, OpenAICompatibleClient)
    assert isinstance(cached_client, CachedLLMClient)


class _CountingClient:
    model = "cache-model"
    reasoning_enabled = False

    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages):
        self.calls += 1
        return LLMResponse(
            content=f"response-{self.calls}",
            reasoning_details=None,
            raw_message={"call": self.calls},
        )


def test_cached_llm_client_context_discriminates_cache(tmp_path):
    cache_dir = tmp_path / "cache"
    messages = [{"role": "user", "content": "hello"}]

    first_inner = _CountingClient()
    first = CachedLLMClient(first_inner, cache_dir, cache_context={"provider": "vllm", "temperature": 0.0})
    first.chat(messages)

    second_inner = _CountingClient()
    second = CachedLLMClient(second_inner, cache_dir, cache_context={"provider": "openrouter", "temperature": 0.0})
    second.chat(messages)

    assert first_inner.calls == 1
    assert second_inner.calls == 1  # different context => cache miss, not reused

    temperature_inner = _CountingClient()
    third = CachedLLMClient(temperature_inner, cache_dir, cache_context={"provider": "vllm", "temperature": 0.7})
    third.chat(messages)
    assert temperature_inner.calls == 1  # different temperature => cache miss


def test_build_llm_client_cache_context_records_provider_endpoint_temperature(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    cached = build_llm_client(
        "vllm",
        model=None,
        base_url="http://example.com/v1",
        api_key=None,
        reasoning_enabled=False,
        temperature=0.3,
        cache_dir=tmp_path / "cache",
    )
    assert isinstance(cached, CachedLLMClient)
    assert cached.cache_context["provider"] == "vllm"
    assert cached.cache_context["endpoint"] == "http://example.com/v1/chat/completions"
    assert cached.cache_context["temperature"] == 0.3


def test_usage_tokens_parses_prompt_and_completion():
    assert _usage_tokens({"prompt_tokens": 12, "completion_tokens": 8}) == (12, 8)
    assert _usage_tokens({"total_tokens": 20}) == (0, 20)
    assert _usage_tokens({}) is None
    assert _usage_tokens(None) is None


class FakeUsageClient:
    model = "usage-model"
    reasoning_enabled = True
    temperature = None
    seed = None
    url = "fake://usage"

    def __init__(self, patch_text: str, usage: dict) -> None:
        self.patch_text = patch_text
        self.usage = usage

    def chat(self, messages):
        return LLMResponse(
            content=f"```diff\n{self.patch_text}\n```",
            reasoning_details=None,
            raw_message={"content": self.patch_text},
            usage=self.usage,
        )


def test_run_llm_search_records_provider_reported_usage_and_provider(tmp_path):
    client = FakeUsageClient(
        (SMOKE_TASK / "candidate.patch").read_text(encoding="utf-8"),
        usage={"prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140},
    )

    result = run_llm_search(SMOKE_TASK, tmp_path / "usage_run", client=client, provider="vllm")

    assert result.status == "promoted"
    assert result.token_source == "provider_usage"
    assert result.prompt_tokens == 100
    assert result.completion_tokens == 40
    assert result.total_tokens == 140
    assert result.records[0].token_source == "provider_usage"
    assert result.records[0].prompt_tokens == 100
    assert result.records[0].completion_tokens == 40

    archive = json.loads((tmp_path / "usage_run" / "candidate_archive.json").read_text(encoding="utf-8"))
    assert archive["run"]["provider"] == "vllm"
    assert archive["run"]["token_source"] == "provider_usage"
    assert archive["run"]["prompt_tokens"] == 100
    assert archive["run"]["completion_tokens"] == 40


def test_run_llm_search_estimates_tokens_when_usage_absent(tmp_path):
    client = FakeClient((SMOKE_TASK / "candidate.patch").read_text(encoding="utf-8"))

    result = run_llm_search(SMOKE_TASK, tmp_path / "estimate_run", client=client, provider="openrouter")

    assert result.token_source == "estimate"
    assert result.total_tokens > 0
    assert result.cost is None


def test_run_llm_search_computes_cost_from_price_schedule(tmp_path):
    client = FakeUsageClient(
        (SMOKE_TASK / "candidate.patch").read_text(encoding="utf-8"),
        usage={"prompt_tokens": 1000, "completion_tokens": 1000, "total_tokens": 2000},
    )
    schedule = {"usage-model": {"input_per_1k": 1.0, "output_per_1k": 2.0, "currency": "USD"}}

    result = run_llm_search(SMOKE_TASK, tmp_path / "cost_run", client=client, price_schedule=schedule)

    assert result.cost == 3.0
    assert result.currency == "USD"
    archive = json.loads((tmp_path / "cost_run" / "candidate_archive.json").read_text(encoding="utf-8"))
    assert archive["run"]["cost"] == 3.0
    assert archive["run"]["currency"] == "USD"


def test_run_llm_search_omits_cost_when_usage_estimated(tmp_path):
    client = FakeClient((SMOKE_TASK / "candidate.patch").read_text(encoding="utf-8"))
    schedule = {"test-model": {"input_per_1k": 1.0, "output_per_1k": 2.0, "currency": "USD"}}

    result = run_llm_search(SMOKE_TASK, tmp_path / "estimate_cost_run", client=client, price_schedule=schedule)

    assert result.token_source == "estimate"
    assert result.cost is None
    archive = json.loads((tmp_path / "estimate_cost_run" / "candidate_archive.json").read_text(encoding="utf-8"))
    assert "cost" not in archive["run"]


def test_parse_generator_metadata_reads_declared_block():
    content = (
        "```diff\ndiff --git a/x b/x\n```\n"
        "```gategrpo-metadata\n"
        '{"intent": "complete_spec_repair", "compatible_routes": ["behavior_repair", "regression_repair"]}\n'
        "```\n"
    )
    metadata = parse_generator_metadata(content)
    assert metadata == GeneratorMetadata(
        intent="complete_spec_repair",
        compatible_routes=("behavior_repair", "regression_repair"),
    )


def test_parse_generator_metadata_returns_none_without_block():
    assert parse_generator_metadata("no metadata here") is None
    assert parse_generator_metadata("```gategrpo-metadata\nnot json\n```") is None


class FakeMetadataClient:
    model = "meta-model"
    reasoning_enabled = True
    temperature = None
    seed = None
    url = "fake://meta"

    def __init__(self, patch_text: str) -> None:
        self.patch_text = patch_text

    def chat(self, messages):
        content = (
            f"```diff\n{self.patch_text}\n```\n"
            "```gategrpo-metadata\n"
            '{"intent": "complete_spec_repair", "compatible_routes": ["behavior_repair"]}\n'
            "```\n"
        )
        return LLMResponse(content=content, reasoning_details=None, raw_message={"content": content})


def test_run_llm_search_records_generator_emitted_metadata(tmp_path):
    client = FakeMetadataClient((SMOKE_TASK / "candidate.patch").read_text(encoding="utf-8"))

    result = run_llm_search(SMOKE_TASK, tmp_path / "meta_run", client=client)

    assert result.status == "promoted"
    record = result.records[0]
    assert record.metadata_source == "generator"
    assert record.candidate_intent == "complete_spec_repair"
    assert record.compatible_routes == ("behavior_repair",)


class FakeRouteMetadataClient:
    model = "route-model"
    reasoning_enabled = True
    temperature = None
    seed = None
    url = "fake://route"

    def __init__(self, patch_text: str, routes: list[str], intent: str = "spec_repair") -> None:
        self.patch_text = patch_text
        self.routes = routes
        self.intent = intent

    def chat(self, messages):
        content = (
            f"```diff\n{self.patch_text}\n```\n"
            "```gategrpo-metadata\n"
            + json.dumps({"intent": self.intent, "compatible_routes": self.routes})
            + "\n```\n"
        )
        return LLMResponse(content=content, reasoning_details=None, raw_message={"content": content})


def test_run_llm_search_drops_unknown_declared_routes(tmp_path):
    client = FakeRouteMetadataClient(
        (SMOKE_TASK / "candidate.patch").read_text(encoding="utf-8"),
        routes=["behavior_repair", "make_it_faster"],
    )

    result = run_llm_search(SMOKE_TASK, tmp_path / "route_run", client=client)

    record = result.records[0]
    assert record.metadata_source == "generator"
    assert record.compatible_routes == ("behavior_repair",)


def test_run_llm_search_falls_back_when_all_declared_routes_unknown(tmp_path):
    client = FakeRouteMetadataClient(
        (SMOKE_TASK / "candidate.patch").read_text(encoding="utf-8"),
        routes=["make_it_faster"],
        intent="",
    )

    result = run_llm_search(SMOKE_TASK, tmp_path / "route_fallback_run", client=client)

    record = result.records[0]
    assert record.metadata_source == "controller"
    assert record.compatible_routes == ()


def test_run_llm_search_records_reproducibility_metadata(tmp_path):
    client = FakeClient((SMOKE_TASK / "candidate.patch").read_text(encoding="utf-8"))

    run_llm_search(SMOKE_TASK, tmp_path / "repro_run", client=client, provider="vllm")

    archive = json.loads(
        (tmp_path / "repro_run" / "candidate_archive.json").read_text(encoding="utf-8")
    )
    repro = archive["run"]["reproducibility"]
    assert repro["provider"] == "vllm"
    assert repro["endpoint"] == "fake://test"
    assert repro["seed"] is None
    assert isinstance(repro["gategrpo_version"], str)
    assert len(repro["config_fingerprint"]) == 64
