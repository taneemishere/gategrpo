import json
from pathlib import Path

import pytest

from gategrpo.policy import run_policy_experiment


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "benchmarks" / "policy_experiment_suite.json"


def test_policy_experiment_selects_on_validation_and_tests_only_selected_policy(tmp_path):
    payload = run_policy_experiment(EXPERIMENT, tmp_path / "policy")

    assert payload["selected_policy"]["id"] == "two_attempt_budgeted"
    assert set(payload["train"]) == {"one_attempt_conservative", "two_attempt_budgeted", "strict_repeat_stop"}
    assert set(payload["validation"]) == {"one_attempt_conservative", "two_attempt_budgeted", "strict_repeat_stop"}
    assert set(payload["test"]) == {"two_attempt_budgeted"}
    assert payload["data_access"]["test"] == "final report only; not used for tuning or selection"
    assert (tmp_path / "policy" / "policy_experiment_results.json").exists()
    assert (tmp_path / "policy" / "policy_experiment_summary.md").exists()


def test_policy_experiment_rejects_overlapping_splits(tmp_path):
    config = json.loads(EXPERIMENT.read_text(encoding="utf-8"))
    config["splits"]["validation"].append(config["splits"]["train"][0])
    config_path = tmp_path / "bad_policy.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="multiple splits"):
        run_policy_experiment(config_path, tmp_path / "policy")
