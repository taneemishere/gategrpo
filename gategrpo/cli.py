from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .benchmark import BASELINES, run_benchmark_suite, run_llm_benchmark_suite, run_llm_model_matrix
from .controller import run_search
from .demo import run_demo
from .generators import generate_candidate_pool
from .init_task import init_task
from .llm import build_llm_client, run_llm_search
from .models import SearchBudget
from .policy import run_policy_experiment
from .report import generate_html_report, generate_showcase_report
from .runner import run_task


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gategrpo", description="Run GateGRPO Phase 1 deterministic gates.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run a task through deterministic gates")
    run_parser.add_argument("task_dir", type=Path)
    run_parser.add_argument("--run-dir", type=Path, default=Path(".gategrpo_runs/latest"))
    run_parser.add_argument("--patch-file", type=Path)
    run_parser.add_argument("--json", action="store_true", help="print machine-readable summary")

    search_parser = subparsers.add_parser("search", help="evaluate candidate patches with archive/routing/stopping")
    search_parser.add_argument("task_dir", type=Path)
    search_parser.add_argument("--run-dir", type=Path, default=Path(".gategrpo_runs/search"))
    search_parser.add_argument("--max-candidates", type=int, default=2)
    search_parser.add_argument("--max-repeated-failures", type=int, default=2)
    search_parser.add_argument("--max-wall-seconds", type=float, default=300.0)
    search_parser.add_argument("--max-writer-tokens", type=int)
    search_parser.add_argument("--max-critic-tokens", type=int)
    search_parser.add_argument("--patch-file", type=Path, action="append", dest="patch_files")
    search_parser.add_argument("--json", action="store_true", help="print machine-readable summary")

    benchmark_parser = subparsers.add_parser("benchmark", help="run a suite across fair baselines")
    benchmark_parser.add_argument("suite_path", type=Path)
    benchmark_parser.add_argument("--run-dir", type=Path, default=Path(".gategrpo_runs/benchmark"))
    benchmark_parser.add_argument("--baseline", choices=BASELINES, action="append", dest="baselines")
    benchmark_parser.add_argument("--repetitions", type=int, default=1)
    benchmark_parser.add_argument("--max-candidates", type=int, default=2)
    benchmark_parser.add_argument("--max-repeated-failures", type=int, default=2)
    benchmark_parser.add_argument("--max-wall-seconds", type=float, default=300.0)
    benchmark_parser.add_argument("--llm-generate", action="store_true", help="generate benchmark candidate pools from a live LLM trace before running baselines")
    benchmark_parser.add_argument("--llm-provider", choices=["openrouter", "vllm", "openai"], default="vllm")
    benchmark_parser.add_argument(
        "--llm-base-url",
        default=None,
        help="base URL for vllm/openai-compatible servers; defaults to $VLLM_BASE_URL or http://localhost:8000/v1 for vllm",
    )
    benchmark_parser.add_argument("--llm-api-key-env", default="VLLM_API_KEY", help="name of the env var to read the API key from; the value is not passed on the command line")
    benchmark_parser.add_argument("--llm-cache-dir", default=None, help="optional on-disk cache for LLM responses; defaults to GATEGRPO_LLM_CACHE_DIR")
    benchmark_parser.add_argument("--llm-temperature", type=float, default=None, help="optional sampling temperature for benchmark generation mode")
    benchmark_parser.add_argument("--llm-seed", type=int, default=None, help="optional sampling seed forwarded to the provider and recorded for reproducibility")
    benchmark_parser.add_argument("--llm-model", default=None, help="LLM model for benchmark generation mode")
    benchmark_parser.add_argument(
        "--llm-reasoning",
        nargs="?",
        const=True,
        type=_parse_bool,
        default=False,
        help="enable reasoning payloads for benchmark generation mode; accepts true/false",
    )
    benchmark_parser.add_argument("--no-llm-reasoning", action="store_false", dest="llm_reasoning", help=argparse.SUPPRESS)
    benchmark_parser.add_argument("--json", action="store_true", help="print machine-readable summary")

    report_parser = subparsers.add_parser("report", help="generate a static HTML report from run artifacts")
    report_parser.add_argument("input_dir", type=Path)
    report_parser.add_argument("--output", type=Path, default=Path(".gategrpo_runs/report.html"))

    policy_parser = subparsers.add_parser("policy-experiment", help="run train/validation/test policy tuning")
    policy_parser.add_argument("config_path", type=Path)
    policy_parser.add_argument("--run-dir", type=Path, default=Path(".gategrpo_runs/policy"))
    policy_parser.add_argument("--json", action="store_true", help="print machine-readable summary")

    demo_parser = subparsers.add_parser("demo", help="run the full local GateGRPO demo")
    demo_parser.add_argument("--run-dir", type=Path, default=Path(".gategrpo_runs/demo"))
    demo_parser.add_argument("--llm", action="store_true", help="generate hero repair candidates with OpenRouter")
    demo_parser.add_argument("--llm-provider", choices=["openrouter", "vllm", "openai"], default="openrouter")
    demo_parser.add_argument(
        "--llm-base-url",
        default=None,
        help="base URL for vllm/openai-compatible servers; defaults to $VLLM_BASE_URL or http://localhost:8000/v1 for vllm",
    )
    demo_parser.add_argument("--llm-api-key-env", default="VLLM_API_KEY", help="name of the env var to read the API key from; the value is not passed on the command line")
    demo_parser.add_argument("--llm-cache-dir", default=None, help="optional on-disk cache for LLM responses; defaults to GATEGRPO_LLM_CACHE_DIR")
    demo_parser.add_argument("--llm-temperature", type=float, default=None, help="optional sampling temperature for --llm mode")
    demo_parser.add_argument("--llm-seed", type=int, default=None, help="optional sampling seed forwarded to the provider and recorded for reproducibility")
    demo_parser.add_argument("--llm-model", default=None, help="LLM model for --llm mode")
    demo_parser.add_argument("--llm-max-candidates", type=int, default=6, help="maximum hero LLM attempts for --llm mode")
    demo_parser.add_argument("--llm-max-repeated-failures", type=int, default=6, help="maximum repeated failures before stopping the hero LLM search")
    demo_parser.add_argument(
        "--llm-reasoning",
        nargs="?",
        const=True,
        type=_parse_bool,
        default=True,
        help="enable OpenRouter reasoning for --llm mode; accepts true/false",
    )
    demo_parser.add_argument("--no-llm-reasoning", action="store_false", dest="llm_reasoning", help=argparse.SUPPRESS)
    demo_parser.add_argument("--json", action="store_true", help="print machine-readable summary")

    init_task_parser = subparsers.add_parser("init-task", help="scaffold or validate a GateGRPO task directory")
    init_task_parser.add_argument("task_dir", type=Path)
    init_task_parser.add_argument("--repo", type=Path, default=None, help="source repository to copy into repo/")
    init_task_parser.add_argument("--instructions", type=Path, default=None, help="instructions markdown file")
    init_task_parser.add_argument("--allowed-path", action="append", dest="allowed_paths", default=None, help="repo-relative path allowed to change")
    init_task_parser.add_argument("--visible-tests", type=Path, default=None, help="optional visible tests directory to copy")
    init_task_parser.add_argument("--regression-tests", type=Path, default=None, help="optional regression tests directory to copy")
    init_task_parser.add_argument("--name", default=None, help="human readable task name")
    init_task_parser.add_argument("--validate", action="store_true", help="validate an existing task without creating files")

    showcase_parser = subparsers.add_parser("showcase", help="run a single-task showcase report")
    showcase_parser.add_argument("task_dir", type=Path)
    showcase_parser.add_argument("--run-dir", type=Path, default=Path(".gategrpo_runs/showcase"))
    showcase_parser.add_argument("--llm", action="store_true", help="generate showcase candidate patches with OpenRouter")
    showcase_parser.add_argument("--llm-provider", choices=["openrouter", "vllm", "openai"], default="openrouter")
    showcase_parser.add_argument(
        "--llm-base-url",
        default=None,
        help="base URL for vllm/openai-compatible servers; defaults to $VLLM_BASE_URL or http://localhost:8000/v1 for vllm",
    )
    showcase_parser.add_argument("--llm-api-key-env", default="VLLM_API_KEY", help="name of the env var to read the API key from; the value is not passed on the command line")
    showcase_parser.add_argument("--llm-cache-dir", default=None, help="optional on-disk cache for LLM responses; defaults to GATEGRPO_LLM_CACHE_DIR")
    showcase_parser.add_argument("--llm-temperature", type=float, default=0.9, help="optional sampling temperature for --llm mode")
    showcase_parser.add_argument("--llm-seed", type=int, default=None, help="optional sampling seed forwarded to the provider and recorded for reproducibility")
    showcase_parser.add_argument("--llm-model", default=None, help="LLM model for --llm mode")
    showcase_parser.add_argument(
        "--llm-reasoning",
        nargs="?",
        const=True,
        type=_parse_bool,
        default=True,
        help="enable OpenRouter reasoning for --llm mode; accepts true/false",
    )
    showcase_parser.add_argument("--no-llm-reasoning", action="store_false", dest="llm_reasoning", help=argparse.SUPPRESS)
    showcase_parser.add_argument("--max-candidates", type=int, default=10)
    showcase_parser.add_argument("--max-repeated-failures", type=int, default=10)

    benchmark_llm_parser = subparsers.add_parser("benchmark-llm", help="run a live LLM benchmark across unique task families")
    benchmark_llm_parser.add_argument("suite_path", type=Path)
    benchmark_llm_parser.add_argument("--run-dir", type=Path, default=Path(".gategrpo_runs/benchmark_llm"))
    benchmark_llm_parser.add_argument("--llm-provider", choices=["openrouter", "vllm", "openai"], default="vllm")
    benchmark_llm_parser.add_argument(
        "--llm-base-url",
        default=None,
        help="base URL for vllm/openai-compatible servers; defaults to $VLLM_BASE_URL or http://localhost:8000/v1 for vllm",
    )
    benchmark_llm_parser.add_argument("--llm-api-key-env", default="VLLM_API_KEY", help="name of the env var to read the API key from; the value is not passed on the command line")
    benchmark_llm_parser.add_argument("--llm-cache-dir", default=None, help="optional on-disk cache for LLM responses; defaults to GATEGRPO_LLM_CACHE_DIR")
    benchmark_llm_parser.add_argument("--llm-temperature", type=float, default=None, help="optional sampling temperature for benchmark-llm mode")
    benchmark_llm_parser.add_argument("--llm-seed", type=int, default=None, help="optional sampling seed forwarded to the provider and recorded for reproducibility")
    benchmark_llm_parser.add_argument(
        "--llm-model",
        action="append",
        dest="llm_model",
        default=None,
        help="LLM model for benchmark-llm mode; pass more than once to run a model-family matrix on the same hard gates",
    )
    benchmark_llm_parser.add_argument(
        "--llm-price-schedule",
        default=None,
        help="path or JSON string mapping model->{input_per_1k,output_per_1k,currency} for cost accounting; defaults to GATEGRPO_LLM_PRICE_SCHEDULE",
    )
    benchmark_llm_parser.add_argument("--repetitions", type=int, default=3)
    benchmark_llm_parser.add_argument("--max-candidates", type=int, default=6)
    benchmark_llm_parser.add_argument("--max-repeated-failures", type=int, default=6)
    benchmark_llm_parser.add_argument(
        "--llm-reasoning",
        nargs="?",
        const=True,
        type=_parse_bool,
        default=False,
        help="enable reasoning payloads for benchmark-llm mode; accepts true/false",
    )
    benchmark_llm_parser.add_argument("--no-llm-reasoning", action="store_false", dest="llm_reasoning", help=argparse.SUPPRESS)
    benchmark_llm_parser.add_argument("--json", action="store_true", help="print machine-readable summary")

    args = parser.parse_args(argv)
    if args.command == "run":
        result = run_task(args.task_dir, args.run_dir, args.patch_file)
        summary = {
            "task_id": result.task_id,
            "status": result.status,
            "workspace": str(result.workspace),
            "trace_path": str(result.trace_path),
            "gates": [{"name": gate.name, "passed": gate.passed, "details": gate.details} for gate in result.gates],
        }
        if args.json:
            print(json.dumps(summary, indent=2, sort_keys=True))
        else:
            print(f"GateGRPO task {result.task_id}: {result.status}")
            print(f"Trace: {result.trace_path}")
            for gate in result.gates:
                marker = "PASS" if gate.passed else "FAIL"
                print(f"  [{marker}] {gate.name}")
        return 0 if result.passed else 1
    if args.command == "benchmark":
        budget = SearchBudget(
            max_candidates=args.max_candidates,
            max_repeated_failures=args.max_repeated_failures,
            max_wall_seconds=args.max_wall_seconds,
        )
        candidate_generator = None
        if args.llm_generate:
            api_key = os.environ.get(args.llm_api_key_env) or None
            client = build_llm_client(
                args.llm_provider,
                model=args.llm_model,
                base_url=args.llm_base_url,
                api_key=api_key,
                reasoning_enabled=args.llm_reasoning,
                temperature=args.llm_temperature,
                seed=args.llm_seed,
                cache_dir=args.llm_cache_dir,
            )

            def candidate_generator(task_dir: Path, task_run_dir: Path):
                return generate_candidate_pool(
                    task_dir,
                    task_run_dir,
                    client=client,
                    budget=budget,
                    provider=args.llm_provider,
                    temperature=args.llm_temperature,
                    seed=args.llm_seed,
                )

        payload = run_benchmark_suite(
            suite_path=args.suite_path,
            run_dir=args.run_dir,
            baselines=tuple(args.baselines) if args.baselines else BASELINES,
            repetitions=args.repetitions,
            budget=budget,
            candidate_generator=candidate_generator,
        )
        if args.json:
            print(json.dumps(payload["summary"], indent=2, sort_keys=True))
        else:
            print(f"GateGRPO benchmark {payload['suite']}")
            print(f"Results: {args.run_dir / 'benchmark_results.json'}")
            print(f"Summary: {args.run_dir / 'benchmark_summary.md'}")
            for baseline, row in payload["summary"].items():
                print(
                    f"  {baseline}: solve@budget={row['solve_at_budget']}"
                    f" avg_attempts={row['avg_attempts']}"
                )
        return 0
    if args.command == "benchmark-llm":
        api_key = os.environ.get(args.llm_api_key_env) or None
        budget = SearchBudget(
            max_candidates=args.max_candidates,
            max_repeated_failures=args.max_repeated_failures,
        )
        models = args.llm_model or []
        if len(models) > 1:
            model_specs = [
                {
                    "id": model,
                    "provider": args.llm_provider,
                    "model": model,
                    "base_url": args.llm_base_url,
                    "api_key": api_key,
                    "reasoning_enabled": args.llm_reasoning,
                    "temperature": args.llm_temperature,
                    "seed": args.llm_seed,
                }
                for model in models
            ]
            result = run_llm_model_matrix(
                suite_path=args.suite_path,
                run_dir=args.run_dir,
                model_specs=model_specs,
                budget=budget,
                repetitions=args.repetitions,
                cache_dir=args.llm_cache_dir,
                price_schedule=args.llm_price_schedule,
            )
            if args.json:
                print(json.dumps(result["matrix"], indent=2, sort_keys=True))
            else:
                print(f"GateGRPO live model matrix {result['suite']}")
                print(f"Results: {args.run_dir / 'model_matrix_results.json'}")
                print(f"Summary: {args.run_dir / 'model_matrix_summary.md'}")
                print(f"Report: {args.run_dir / 'model_matrix_report.html'}")
                for entry in result["matrix"]:
                    overall = entry.get("overall", {})
                    print(
                        f"  {entry['model']} ({entry['provider']}):"
                        f" solve@budget={overall.get('solve_at_budget')}"
                        f" avg_attempts={overall.get('avg_attempts')}"
                    )
            return 0
        payload = run_llm_benchmark_suite(
            suite_path=args.suite_path,
            run_dir=args.run_dir,
            provider=args.llm_provider,
            base_url=args.llm_base_url,
            model=models[0] if models else None,
            api_key=api_key,
            reasoning_enabled=args.llm_reasoning,
            temperature=args.llm_temperature,
            seed=args.llm_seed,
            budget=budget,
            repetitions=args.repetitions,
            cache_dir=args.llm_cache_dir,
            price_schedule=args.llm_price_schedule,
        )
        if args.json:
            print(json.dumps(payload["summary"], indent=2, sort_keys=True))
        else:
            print(f"GateGRPO live benchmark {payload['suite']}")
            print(f"Results: {args.run_dir / 'llm_benchmark_results.json'}")
            print(f"Summary: {args.run_dir / 'llm_benchmark_summary.md'}")
            for task_id, row in payload["summary"].items():
                print(
                    f"  {task_id}: solve@budget={row['solve_at_budget']}"
                    f" avg_attempts={row['avg_attempts']}"
                )
        return 0
    if args.command == "report":
        output = generate_html_report(args.input_dir, args.output)
        print(f"GateGRPO report: {output}")
        return 0
    if args.command == "policy-experiment":
        payload = run_policy_experiment(args.config_path, args.run_dir)
        summary = {
            "experiment": payload["experiment"],
            "selected_policy": payload["selected_policy"]["id"],
            "test": payload["test"],
            "results_path": str(args.run_dir / "policy_experiment_results.json"),
            "summary_path": str(args.run_dir / "policy_experiment_summary.md"),
        }
        if args.json:
            print(json.dumps(summary, indent=2, sort_keys=True))
        else:
            print(f"GateGRPO policy experiment {payload['experiment']}")
            print(f"Selected policy: {payload['selected_policy']['id']}")
            print(f"Results: {args.run_dir / 'policy_experiment_results.json'}")
            print(f"Summary: {args.run_dir / 'policy_experiment_summary.md'}")
        return 0
    if args.command == "demo":
        try:
            api_key = os.environ.get(args.llm_api_key_env) or None
            summary = run_demo(
                args.run_dir,
                use_llm=args.llm,
                llm_provider=args.llm_provider,
                llm_model=args.llm_model,
                llm_base_url=args.llm_base_url,
                llm_api_key=api_key,
                llm_cache_dir=args.llm_cache_dir,
                llm_temperature=args.llm_temperature,
                llm_seed=args.llm_seed,
                llm_reasoning=args.llm_reasoning,
                llm_max_candidates=args.llm_max_candidates,
                llm_max_repeated_failures=args.llm_max_repeated_failures,
            )
        except RuntimeError as exc:
            print(f"GateGRPO demo failed: {exc}", file=sys.stderr)
            return 2
        if args.json:
            print(json.dumps(summary, indent=2, sort_keys=True))
        else:
            hero = summary["hero"]
            policy = summary["policy"]
            full = summary["benchmark"]["summary"]["full_gategrpo"]
            single = summary["benchmark"]["summary"]["single_shot"]
            print("GateGRPO demo complete")
            print(f"LLM required: {'yes' if summary['llm_required'] else 'no'}")
            print(f"Mode: {summary['llm_mode']}")
            if summary["llm_model"]:
                print(f"LLM model: {summary['llm_model']}")
                print(f"LLM reasoning: {'yes' if summary['llm_reasoning'] else 'no'}")
            print(f"Hero search: {hero['status']} after {hero['attempts']} attempts")
            final_state = "promoted" if hero["status"] == "promoted" else hero["stop_reason"]
            print(f"Repair path: {hero['first_failure'] or 'direct'} -> {hero['final_route']} -> {final_state}")
            print(f"Hero report: {hero['report']}")
            print(
                "Benchmark: "
                f"single_shot solve@budget={single['solve_at_budget']} vs "
                f"full_gategrpo solve@budget={full['solve_at_budget']}"
            )
            print(f"Benchmark report: {summary['benchmark']['report']}")
            print(f"Policy selected: {policy['selected_policy']}")
            print(f"Policy summary: {policy['summary']}")
            print(f"Demo summary: {args.run_dir / 'demo_summary.json'}")
        return 0
    if args.command == "init-task":
        result = init_task(
            args.task_dir,
            repo=args.repo,
            instructions=args.instructions,
            allowed_paths=tuple(args.allowed_paths or ()),
            visible_tests=args.visible_tests,
            regression_tests=args.regression_tests,
            name=args.name,
            validate=args.validate,
        )
        _print_init_task_report(result)
        if result.passed:
            print(
                f"Next: python3 -m gategrpo showcase {args.task_dir} --llm --llm-provider vllm "
                "--llm-base-url <your-openai-compatible-base-url> --run-dir .gategrpo_runs/showcase"
            )
            print(f"Offline: python3 -m gategrpo showcase {args.task_dir}")
            return 0
        return 1
    if args.command == "showcase":
        api_key = os.environ.get(args.llm_api_key_env) or None
        budget = SearchBudget(
            max_candidates=args.max_candidates,
            max_repeated_failures=args.max_repeated_failures,
        )
        if args.llm:
            result = run_llm_search(
                args.task_dir,
                args.run_dir,
                budget=budget,
                model=args.llm_model,
                reasoning_enabled=args.llm_reasoning,
                provider=args.llm_provider,
                base_url=args.llm_base_url,
                api_key=api_key,
                cache_dir=args.llm_cache_dir,
                temperature=args.llm_temperature,
                seed=args.llm_seed,
            )
        else:
            result = run_search(args.task_dir, args.run_dir, budget)
        report = generate_showcase_report(args.run_dir, args.run_dir / "showcase_report.html")
        archive_path = args.run_dir / "candidate_archive.json"
        archive = json.loads(archive_path.read_text(encoding="utf-8"))
        run_header = archive.get("run", {})
        print("GateGRPO showcase complete")
        print(f"Task: {run_header.get('task_name', result.task_id)}")
        print(f"Status: {result.status}")
        print(f"Attempts: {len(result.records)}")
        print(f"Repair path: {run_header.get('repair_path', 'unknown')}")
        print(f"Report: {report}")
        print(f"Archive: {archive_path}")
        return 0
    if args.command == "search":
        budget = SearchBudget(
            max_candidates=args.max_candidates,
            max_repeated_failures=args.max_repeated_failures,
            max_wall_seconds=args.max_wall_seconds,
            max_writer_tokens=args.max_writer_tokens,
            max_critic_tokens=args.max_critic_tokens,
        )
        result = run_search(args.task_dir, args.run_dir, budget, args.patch_files)
        summary = {
            "task_id": result.task_id,
            "status": result.status,
            "stop_reason": result.stop_reason,
            "trace_path": str(result.trace_path),
            "archive_path": str(result.archive_path),
            "records": [
                {
                    "candidate_id": record.candidate_id,
                    "status": record.status,
                    "route": record.route,
                    "failure_type": record.failure_type,
                    "failure_fingerprint": record.failure_fingerprint,
                    "parent_id": record.parent_id,
                    "generation": record.generation,
                    "score": record.score,
                    "evidence_summary": record.evidence.summary if record.evidence else None,
                    "evidence_id": record.evidence.evidence_id if record.evidence else None,
                }
                for record in result.records
            ],
        }
        if args.json:
            print(json.dumps(summary, indent=2, sort_keys=True))
        else:
            print(f"GateGRPO search {result.task_id}: {result.status} ({result.stop_reason})")
            print(f"Trace: {result.trace_path}")
            print(f"Archive: {result.archive_path}")
            for record in result.records:
                print(
                    f"  {record.candidate_id}: {record.status}"
                    f" route={record.route} failure={record.failure_type or 'none'}"
                )
        return 0 if result.passed else 1
    return 2


def _print_init_task_report(result) -> None:
    action = "Validated" if result.validated_only else "Created"
    print(f"GateGRPO init-task {action}: {result.task_dir}")
    for check in result.checks:
        marker = "✓" if check.passed else "✗"
        print(f"  {marker} {check.name}: {check.detail}")
    for note in result.notes:
        print(f"  i {note}")


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")
