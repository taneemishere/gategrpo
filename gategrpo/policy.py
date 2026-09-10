from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from .benchmark import run_benchmark_suite
from .models import SearchBudget


def run_policy_experiment(config_path: Path, run_dir: Path) -> dict[str, Any]:
    config = _read_json(config_path)
    source_suite_path = _resolve_path(config_path, config["source_suite"])
    source_suite = _read_json(source_suite_path)
    cases_by_id = {case["id"]: case for case in source_suite["cases"]}
    splits = _validate_splits(config["splits"], cases_by_id)
    policies = config["policies"]
    run_dir.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix="gategrpo_policy_") as temp_dir:
        temp_root = Path(temp_dir)
        split_suite_paths = {
            split_name: _write_split_suite(
                source_suite_path=source_suite_path,
                source_suite=source_suite,
                cases=[cases_by_id[case_id] for case_id in case_ids],
                split_name=split_name,
                output_dir=temp_root,
            )
            for split_name, case_ids in splits.items()
        }

        train_results = _run_policies(split_suite_paths["train"], run_dir / "train", policies)
        validation_results = _run_policies(split_suite_paths["validation"], run_dir / "validation", policies)
        selected_policy = _select_policy(validation_results, policies)
        test_results = _run_policies(
            split_suite_paths["test"],
            run_dir / "test",
            [selected_policy],
        )

    payload = {
        "experiment": config["name"],
        "source_suite": str(source_suite_path),
        "data_access": {
            "train": "policy tuning",
            "validation": "policy selection",
            "test": "final report only; not used for tuning or selection",
        },
        "splits": splits,
        "policies": policies,
        "train": train_results,
        "validation": validation_results,
        "selected_policy": selected_policy,
        "test": test_results,
    }
    (run_dir / "policy_experiment_results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (run_dir / "policy_experiment_summary.md").write_text(_summary_markdown(payload), encoding="utf-8")
    return payload


def _run_policies(suite_path: Path, run_dir: Path, policies: list[dict[str, Any]]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for policy in policies:
        budget = SearchBudget(
            max_candidates=int(policy["max_candidates"]),
            max_repeated_failures=int(policy["max_repeated_failures"]),
        )
        payload = run_benchmark_suite(
            suite_path=suite_path,
            run_dir=run_dir / policy["id"],
            baselines=("full_gategrpo",),
            repetitions=1,
            budget=budget,
        )
        summary = payload["summary"]["full_gategrpo"]
        results[policy["id"]] = {
            "summary": summary,
            "score": _score(summary),
            "results_path": str(run_dir / policy["id"] / "benchmark_results.json"),
        }
    return results


def _select_policy(validation_results: dict[str, Any], policies: list[dict[str, Any]]) -> dict[str, Any]:
    policies_by_id = {policy["id"]: policy for policy in policies}
    best_id = sorted(
        validation_results,
        key=lambda policy_id: (
            validation_results[policy_id]["score"],
            validation_results[policy_id]["summary"]["solve_at_budget"],
            -validation_results[policy_id]["summary"]["avg_attempts"],
        ),
        reverse=True,
    )[0]
    return policies_by_id[best_id]


def _score(summary: dict[str, Any]) -> float:
    return round(
        summary["solve_at_budget"] * 100
        - summary["avg_attempts"]
        - summary["unsolved_task_cost"]
        - summary["repeated_failure_rate"] * 5,
        6,
    )


def _validate_splits(splits: dict[str, list[str]], cases_by_id: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    required = {"train", "validation", "test"}
    if set(splits) != required:
        raise ValueError("policy experiment requires train, validation, and test splits")
    seen: set[str] = set()
    normalized: dict[str, list[str]] = {}
    for split_name in ("train", "validation", "test"):
        case_ids = list(splits[split_name])
        if not case_ids:
            raise ValueError(f"{split_name} split is empty")
        overlap = seen.intersection(case_ids)
        if overlap:
            raise ValueError(f"case ids appear in multiple splits: {sorted(overlap)}")
        missing = [case_id for case_id in case_ids if case_id not in cases_by_id]
        if missing:
            raise ValueError(f"unknown case ids in {split_name}: {missing}")
        seen.update(case_ids)
        normalized[split_name] = case_ids
    return normalized


def _write_split_suite(
    source_suite_path: Path,
    source_suite: dict[str, Any],
    cases: list[dict[str, Any]],
    split_name: str,
    output_dir: Path,
) -> Path:
    repo_root = source_suite_path.resolve().parent.parent
    suite = {
        "name": f"{source_suite['name']}_{split_name}",
        "cases": [_absolutize_case(case, repo_root) for case in cases],
    }
    path = output_dir / f"{split_name}.json"
    path.write_text(json.dumps(suite, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _absolutize_case(case: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    copied = dict(case)
    task_dir = Path(str(copied["task_dir"]))
    if not task_dir.is_absolute():
        copied["task_dir"] = str(repo_root / task_dir)
    return copied


def _summary_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# GateGRPO policy experiment: {payload['experiment']}",
        "",
        "Data boundary:",
        "",
        "- Train: policy tuning",
        "- Validation: policy selection",
        "- Test: final report only; selected policy only",
        "",
        f"Selected policy: `{payload['selected_policy']['id']}`",
        "",
        "## Validation scores",
        "",
        "| policy | score | solve@budget | avg attempts | unsolved cost |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for policy_id, row in sorted(payload["validation"].items()):
        summary = row["summary"]
        lines.append(
            f"| {policy_id} | {row['score']} | {summary['solve_at_budget']} | "
            f"{summary['avg_attempts']} | {summary['unsolved_task_cost']} |"
        )
    lines.extend(["", "## Held-out test", "", "| policy | score | solve@budget | avg attempts |", "| --- | ---: | ---: | ---: |"])
    for policy_id, row in sorted(payload["test"].items()):
        summary = row["summary"]
        lines.append(f"| {policy_id} | {row['score']} | {summary['solve_at_budget']} | {summary['avg_attempts']} |")
    lines.append("")
    return "\n".join(lines)


def _resolve_path(config_path: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    config_relative = config_path.resolve().parent.parent / path
    if config_relative.exists():
        return config_relative
    cwd_relative = Path.cwd() / path
    if cwd_relative.exists():
        return cwd_relative
    return Path(__file__).resolve().parents[1] / path


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)
