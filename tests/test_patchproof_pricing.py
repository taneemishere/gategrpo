import json

from patchproof.pricing import (
    PRICE_SCHEDULE_ENV,
    estimate_cost,
    load_price_schedule,
    schedule_currency,
)


SCHEDULE = {
    "demo-model": {"input_per_1k": 0.5, "output_per_1k": 1.5, "currency": "USD"},
}


def test_estimate_cost_uses_input_and_output_prices():
    cost = estimate_cost("demo-model", prompt_tokens=2000, completion_tokens=1000, schedule=SCHEDULE)
    assert cost == 2.5


def test_estimate_cost_returns_none_for_unknown_model():
    assert estimate_cost("missing", 100, 100, SCHEDULE) is None


def test_estimate_cost_returns_none_when_no_prices_present():
    assert estimate_cost("bare", 100, 100, {"bare": {"currency": "USD"}}) is None


def test_schedule_currency():
    assert schedule_currency("demo-model", SCHEDULE) == "USD"
    assert schedule_currency("missing", SCHEDULE) is None


def test_load_price_schedule_from_json_string():
    schedule = load_price_schedule(json.dumps(SCHEDULE))
    assert schedule["demo-model"]["output_per_1k"] == 1.5


def test_load_price_schedule_from_file(tmp_path):
    path = tmp_path / "prices.json"
    path.write_text(json.dumps(SCHEDULE), encoding="utf-8")
    schedule = load_price_schedule(path)
    assert schedule == SCHEDULE


def test_load_price_schedule_from_env(monkeypatch, tmp_path):
    path = tmp_path / "prices.json"
    path.write_text(json.dumps(SCHEDULE), encoding="utf-8")
    monkeypatch.setenv(PRICE_SCHEDULE_ENV, str(path))
    assert load_price_schedule() == SCHEDULE


def test_load_price_schedule_missing_returns_empty():
    assert load_price_schedule(None) == {}
    assert load_price_schedule("not valid json {") == {}
