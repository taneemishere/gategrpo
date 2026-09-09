import json
from pathlib import Path
from types import SimpleNamespace

from patchproof import demo as demo_module
from patchproof.llm import VLLM_DEFAULT_MODEL


def test_run_demo_writes_summary_without_requiring_llm(tmp_path, monkeypatch):
    def fake_search(task_dir, run_dir):
        return SimpleNamespace(
            status="promoted",
            stop_reason="promoted",
            archive_path=run_dir / "candidate_archive.json",
            records=[
                SimpleNamespace(failure_type="visible_tests", route="initial_repair"),
                SimpleNamespace(failure_type=None, route="behavior_repair"),
            ],
        )

    def fake_report(input_dir: Path, output_path: Path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("<html></html>", encoding="utf-8")
        return output_path

    def fake_benchmark(suite_path, run_dir):
        return {
            "summary": {
                "single_shot": {"solve_at_budget": 0.2},
                "full_patchproof": {"solve_at_budget": 0.8},
            }
        }

    def fake_policy(config_path, run_dir):
        return {"selected_policy": {"id": "two_attempt_budgeted"}}

    monkeypatch.setattr(demo_module, "run_search", fake_search)
    monkeypatch.setattr(demo_module, "generate_html_report", fake_report)
    monkeypatch.setattr(demo_module, "run_benchmark_suite", fake_benchmark)
    monkeypatch.setattr(demo_module, "run_policy_experiment", fake_policy)

    summary = demo_module.run_demo(tmp_path / "demo")

    assert summary["llm_required"] is False
    assert summary["hero"]["first_failure"] == "visible_tests"
    assert summary["hero"]["final_route"] == "behavior_repair"
    saved = json.loads((tmp_path / "demo" / "demo_summary.json").read_text(encoding="utf-8"))
    assert saved["policy"]["selected_policy"] == "two_attempt_budgeted"


def test_run_demo_can_use_llm_for_hero_search(tmp_path, monkeypatch):
    seen_calls = []

    def fake_llm_search(
        task_dir,
        run_dir,
        budget,
        model,
        reasoning_enabled,
        temperature=None,
        seed=None,
        provider="openrouter",
        base_url=None,
        api_key=None,
        cache_dir=None,
    ):
        seen_calls.append(
            {
                "provider": provider,
                "base_url": base_url,
                "api_key": api_key,
                "model": model,
                "reasoning_enabled": reasoning_enabled,
                "temperature": temperature,
                "budget": {
                    "max_candidates": budget.max_candidates,
                    "max_repeated_failures": budget.max_repeated_failures,
                },
            }
        )
        return SimpleNamespace(
            status="promoted",
            stop_reason="promoted",
            archive_path=run_dir / "candidate_archive.json",
            records=[SimpleNamespace(failure_type=None, route="initial_repair")],
        )

    def fake_report(input_dir: Path, output_path: Path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("<html></html>", encoding="utf-8")
        return output_path

    def fake_benchmark(suite_path, run_dir):
        return {
            "summary": {
                "single_shot": {"solve_at_budget": 0.2},
                "full_patchproof": {"solve_at_budget": 0.8},
            }
        }

    def fake_policy(config_path, run_dir):
        return {"selected_policy": {"id": "two_attempt_budgeted"}}

    monkeypatch.setattr(demo_module, "run_llm_search", fake_llm_search)
    monkeypatch.setattr(demo_module, "generate_html_report", fake_report)
    monkeypatch.setattr(demo_module, "run_benchmark_suite", fake_benchmark)
    monkeypatch.setattr(demo_module, "run_policy_experiment", fake_policy)

    summary = demo_module.run_demo(
        tmp_path / "demo",
        use_llm=True,
        llm_model="test-model",
        llm_reasoning=False,
        llm_max_candidates=4,
        llm_max_repeated_failures=5,
    )

    assert summary["llm_required"] is True
    assert summary["llm_mode"] == "openrouter"
    assert summary["llm_model"] == "test-model"
    assert summary["llm_reasoning"] is False
    assert summary["hero"]["final_route"] == "initial_repair"
    assert seen_calls == [
        {
            "provider": "openrouter",
            "base_url": None,
            "api_key": None,
            "model": "test-model",
            "reasoning_enabled": False,
            "temperature": None,
            "budget": {"max_candidates": 4, "max_repeated_failures": 5},
        }
    ]


def test_run_demo_can_use_vllm_for_hero_search(tmp_path, monkeypatch):
    seen_calls = []

    def fake_llm_search(
        task_dir,
        run_dir,
        budget,
        model,
        reasoning_enabled,
        temperature=None,
        seed=None,
        provider="openrouter",
        base_url=None,
        api_key=None,
        cache_dir=None,
    ):
        seen_calls.append(
            {
                "provider": provider,
                "base_url": base_url,
                "api_key": api_key,
                "model": model,
                "reasoning_enabled": reasoning_enabled,
                "temperature": temperature,
                "budget": {
                    "max_candidates": budget.max_candidates,
                    "max_repeated_failures": budget.max_repeated_failures,
                },
            }
        )
        return SimpleNamespace(
            status="promoted",
            stop_reason="promoted",
            archive_path=run_dir / "candidate_archive.json",
            records=[SimpleNamespace(failure_type=None, route="initial_repair")],
        )

    def fake_report(input_dir: Path, output_path: Path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("<html></html>", encoding="utf-8")
        return output_path

    def fake_benchmark(suite_path, run_dir):
        return {
            "summary": {
                "single_shot": {"solve_at_budget": 0.2},
                "full_patchproof": {"solve_at_budget": 0.8},
            }
        }

    def fake_policy(config_path, run_dir):
        return {"selected_policy": {"id": "two_attempt_budgeted"}}

    monkeypatch.setattr(demo_module, "run_llm_search", fake_llm_search)
    monkeypatch.setattr(demo_module, "generate_html_report", fake_report)
    monkeypatch.setattr(demo_module, "run_benchmark_suite", fake_benchmark)
    monkeypatch.setattr(demo_module, "run_policy_experiment", fake_policy)

    summary = demo_module.run_demo(
        tmp_path / "demo_vllm",
        use_llm=True,
        llm_provider="vllm",
        llm_base_url="http://localhost:8000/v1",
        llm_reasoning=False,
        llm_max_candidates=7,
        llm_max_repeated_failures=8,
    )

    assert summary["llm_required"] is True
    assert summary["llm_mode"] == "vllm"
    assert summary["llm_model"] == VLLM_DEFAULT_MODEL
    assert summary["llm_base_url"] == "http://localhost:8000/v1"
    assert seen_calls == [
        {
            "provider": "vllm",
            "base_url": "http://localhost:8000/v1",
            "api_key": None,
            "model": None,
            "reasoning_enabled": False,
            "temperature": None,
            "budget": {"max_candidates": 7, "max_repeated_failures": 8},
        }
    ]
