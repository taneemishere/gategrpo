"""Inspect generated SEARCH/REPLACE patches against the SFT-merged model.

Generates one or two candidates per pilot task, runs the verifier, and reports
why patch_applies fails by comparing the generated SEARCH text to the actual
file content.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import uuid
from pathlib import Path

from ..llm import _initial_prompt, build_llm_client
from ..runner import run_task
from ..task import load_task
from .rewards import _aider_blocks


def _format_search_preview(search: str, width: int = 80) -> str:
    lines = search.splitlines()[:8]
    out = "\n".join(lines)
    if len(lines) < search.count("\n"):
        out += "\n..."
    return out


def main() -> int:
    suite_path = Path("benchmarks/rl_pilot_suite.json")
    client = build_llm_client(
        "vllm",
        model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        base_url="http://localhost:8000/v1",
        api_key=None,
        reasoning_enabled=False,
        temperature=1.0,
    )

    with suite_path.open(encoding="utf-8") as f:
        suite = json.load(f)

    for case in suite["cases"]:
        task_dir = Path(case["task_dir"])
        task = load_task(task_dir)
        prompt_text = _initial_prompt(task_dir)
        messages = [{"role": "user", "content": prompt_text}]

        for sample in range(2):
            response = client.chat(messages)
            patch_text = response.content or ""

            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", suffix=".patch", delete=False
            ) as f:
                f.write(patch_text)
                patch_path = Path(f.name)

            run_root = Path(".gategrpo_runs/inspect")
            run_root.mkdir(parents=True, exist_ok=True)
            run_dir = run_root / f"{task_dir.name}_{uuid.uuid4().hex[:8]}"
            run_dir.mkdir(parents=True, exist_ok=True)

            try:
                result = run_task(task_dir, run_dir, patch_path)
                failed_gate = None
                for gate in result.gates:
                    if not gate.passed:
                        failed_gate = gate.name
                        break
                promoted = result.passed
            finally:
                patch_path.unlink(missing_ok=True)
                shutil.rmtree(run_dir, ignore_errors=True)

            print(f"\n{'=' * 70}")
            print(f"task={task_dir.name} sample={sample + 1}")
            print(f"promoted={promoted} first_failed_gate={failed_gate}")
            print(f"completion_length={len(patch_text)} chars")
            print("--- completion preview ---")
            print(patch_text[:800].replace("\r", ""))
            print("--- end preview ---")

            blocks = _aider_blocks(patch_text)
            if not blocks:
                print("NO AIDER BLOCKS PARSED")
                continue

            for path, search, replace in blocks:
                print(f"\n  block: {path}  allowed={path in task.allowed_paths}")
                workspace = task_dir / "repo"
                file_path = workspace / path
                if not file_path.exists():
                    print(f"    FILE DOES NOT EXIST")
                    continue

                file_text = file_path.read_text(encoding="utf-8")
                exact_match = search in file_text

                import difflib

                matcher = difflib.SequenceMatcher(None, search, file_text)
                matching = sum(b.size for b in matcher.get_matching_blocks())
                similarity = matching / len(search) if search else 0.0

                print(f"    search_len={len(search)}  exact_match={exact_match}  fuzzy_similarity={similarity:.3f}")
                print(f"    search preview:\n{_format_search_preview(search)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
