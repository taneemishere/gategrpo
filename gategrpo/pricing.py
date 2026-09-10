from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

PRICE_SCHEDULE_ENV = "GATEGRPO_LLM_PRICE_SCHEDULE"


def load_price_schedule(source: str | Path | None = None) -> dict[str, Any]:
    """Load a model price schedule.

    The schedule maps a model name to per-1K-token prices, for example::

        {
          "Qwen3-Coder-30B-A3B-Instruct": {
            "input_per_1k": 0.0,
            "output_per_1k": 0.0,
            "currency": "USD"
          }
        }

    ``source`` may be a path or JSON string. When omitted, the
    ``GATEGRPO_LLM_PRICE_SCHEDULE`` env var is consulted. A missing or
    unreadable schedule yields an empty mapping so cost accounting is simply
    skipped rather than failing a research run.
    """
    resolved = source if source is not None else os.environ.get(PRICE_SCHEDULE_ENV)
    if not resolved:
        return {}
    text = _read_schedule_text(resolved)
    if text is None:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _read_schedule_text(resolved: str | Path) -> str | None:
    candidate = Path(resolved) if not isinstance(resolved, Path) else resolved
    try:
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")
    except OSError:
        return None
    if isinstance(resolved, str):
        return resolved
    return None


def estimate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    schedule: dict[str, Any],
) -> float | None:
    """Return the currency cost for a call, or ``None`` when no price is known."""
    entry = schedule.get(model)
    if not isinstance(entry, dict):
        return None
    input_price = _price(entry.get("input_per_1k"))
    output_price = _price(entry.get("output_per_1k"))
    if input_price is None and output_price is None:
        return None
    cost = 0.0
    if input_price is not None:
        cost += (prompt_tokens / 1000.0) * input_price
    if output_price is not None:
        cost += (completion_tokens / 1000.0) * output_price
    return round(cost, 6)


def schedule_currency(model: str, schedule: dict[str, Any]) -> str | None:
    entry = schedule.get(model)
    if isinstance(entry, dict):
        currency = entry.get("currency")
        if isinstance(currency, str):
            return currency
    return None


def _price(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None
