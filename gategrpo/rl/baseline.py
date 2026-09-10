"""Base-model LLM baseline runner for the GateGRPO-RL pilot suite.

This script does not train anything.  It is intended to be executed after a
Qwen2.5-Coder-1.5B model is served through vLLM.  It records the pre-training
``solve@budget``, attempts, tokens, and cost so the GRPO/DPO runs have a
reference to beat.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from ..benchmark import run_llm_benchmark_suite
from ..models import SearchBudget


DEFAULT_SUITE = (
    Path(__file__).resolve().parents[2] / "benchmarks" / "rl_pilot_suite.json"
)
DEFAULT_RUN_DIR = Path(".gategrpo_runs/pilot_baseline")
DEFAULT_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gategrpo-rl-baseline",
        description="Run the base-model LLM baseline on the RL pilot suite.",
    )
    parser.add_argument(
        "--suite",
        type=Path,
        default=DEFAULT_SUITE,
        help="Path to the benchmark suite JSON.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="Directory to write baseline results.",
    )
    parser.add_argument(
        "--llm-provider",
        default="vllm",
        choices=["vllm", "openai", "openrouter"],
        help="Provider for the LLM endpoint.",
    )
    parser.add_argument(
        "--llm-base-url",
        default=DEFAULT_BASE_URL,
        help="OpenAI-compatible endpoint URL.",
    )
    parser.add_argument(
        "--llm-model",
        default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        help="Model name as exposed by the endpoint.",
    )
    parser.add_argument(
        "--llm-api-key-env",
        default="VLLM_API_KEY",
        help="Environment variable holding the API key, if any.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
        help="Number of independent repair attempts per task.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=6,
        help="Max candidate patches per repair search.",
    )
    parser.add_argument(
        "--max-repeated-failures",
        type=int,
        default=6,
        help="Max repeated failures before the search stops.",
    )

    args = parser.parse_args(argv)
    api_key = os.environ.get(args.llm_api_key_env) or None

    budget = SearchBudget(
        max_candidates=args.max_candidates,
        max_repeated_failures=args.max_repeated_failures,
    )

    payload = run_llm_benchmark_suite(
        suite_path=args.suite,
        run_dir=args.run_dir,
        provider=args.llm_provider,
        base_url=args.llm_base_url,
        model=args.llm_model,
        api_key=api_key,
        budget=budget,
        repetitions=args.repetitions,
    )

    print(f"Pilot baseline complete: {args.suite}")
    print(f"  Run dir: {args.run_dir}")
    print(f"  JSON summary: {args.run_dir / 'llm_benchmark_summary.json'}")
    print(f"  Markdown report: {args.run_dir / 'llm_benchmark_summary.md'}")

    overall = payload["summary"].get("all_tasks", {})
    print(
        f"  solve@budget={overall.get('solve_at_budget', 'n/a')} "
        f"avg_attempts={overall.get('avg_attempts', 'n/a')} "
        f"avg_tokens={overall.get('avg_tokens', 'n/a')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
