"""Verifiable, hard-gate reward functions for GateGRPO-RL.

The reward model is the GateGRPO gate pipeline itself.  No learned reward
model is required; the signal is fully grounded in execution and the structured
gate taxonomy that GateGRPO already computes.
"""

from __future__ import annotations

import difflib
import re
from pathlib import Path
from typing import Any

from ..models import GateResult, RunResult

# Partial-credit weights for the pre-release gates.  The release gate is
# intentionally excluded from partial credit and represented by the terminal
# promotion reward, so a non-promoted patch cannot accidentally score close to
# a promoted one.
GATE_PARTIAL_WEIGHTS: dict[str, float] = {
    "scope_guard": 0.05,
    "patch_applies": 0.05,
    "python_ast": 0.10,
    "secret_scan": 0.10,
    "visible_tests": 0.20,
}

# The terminal outcome reward.  It dominates the partial-credit term so the
# policy optimizes for promotion, not just clearing cheap syntactic gates.
PROMOTION_REWARD: float = 3.0

# Scaling factor for the generation-length penalty.
COMPLETION_LENGTH_PENALTY_SCALE: float = -0.05

# Small scaling factor for the dollar-cost penalty (if cost reporting is on).
COST_PENALTY_SCALE: float = -0.01

# Small format-shaping reward constants.  These are intentionally smaller than
# the gate partial credits so they guide early exploration without dominating
# the true execution signal once the policy reaches the executable manifold.
FORMAT_BLOCK_REWARD: float = 0.025
FORMAT_FUZZY_SCALE: float = 0.10


def gate_partial_credit(gate: GateResult) -> float:
    """Return the partial-credit reward for a single pre-release gate."""
    if not gate.passed or gate.name == "release_gate_regressions":
        return 0.0
    return GATE_PARTIAL_WEIGHTS.get(gate.name, 0.0)


def gategrpo_reward(
    gates: list[GateResult],
    promoted: bool,
    completion_tokens: int = 0,
    max_tokens: int = 1,
    cost: float | None = None,
    *,
    length_penalty_scale: float = COMPLETION_LENGTH_PENALTY_SCALE,
    promotion_reward: float = PROMOTION_REWARD,
    cost_penalty_scale: float = COST_PENALTY_SCALE,
) -> float:
    """Compute a scalar reward from a GateGRPO gate run.

    Args:
        gates: The list of ``GateResult`` objects returned by ``run_task``.
        promoted: Whether the patch was promoted (passed every gate).
        completion_tokens: Number of tokens in the generated completion.
        max_tokens: The max-allowed completion length, used to normalise the
            length penalty.  Use ``1`` if no explicit cap is available.
        cost: Optional provider-reported dollar cost of the generation.
        length_penalty_scale: Weight for the length penalty term.
        promotion_reward: The scalar reward for a promoted patch.
        cost_penalty_scale: Weight for the dollar-cost penalty term.

    Returns:
        A scalar reward.  Promoted patches are clearly separated from
        non-promoted ones because the ``promotion_reward`` dominates.
    """
    # 1. Partial credit for pre-release progress.
    partial = sum(gate_partial_credit(g) for g in gates)

    # 2. Terminal promotion signal.
    outcome = promotion_reward if promoted else 0.0

    # 3. Penalise overly long generations to keep token cost in check.
    token_frac = completion_tokens / max(max_tokens, 1)
    length_penalty = length_penalty_scale * token_frac

    # 4. Optional cost shaping.
    cost_penalty = 0.0
    if cost is not None:
        cost_penalty = cost_penalty_scale * cost

    return partial + outcome + length_penalty + cost_penalty


def reward_from_run_result(
    result: RunResult,
    completion_tokens: int = 0,
    max_tokens: int = 1,
    cost: float | None = None,
    **kwargs: Any,
) -> float:
    """Convenience wrapper that turns a ``RunResult`` into a reward."""
    return gategrpo_reward(
        gates=result.gates,
        promoted=result.passed,
        completion_tokens=completion_tokens,
        max_tokens=max_tokens,
        cost=cost,
        **kwargs,
    )


def _aider_blocks(patch_text: str) -> list[tuple[str, str, str]]:
    """Parse Aider-style SEARCH/REPLACE blocks from a completion.

    Returns a list of (path, search_text, replace_text) tuples.
    """
    # Strip common markdown code fences.
    text = re.sub(r"```[\w-]*\s*\n", "\n", patch_text)
    text = re.sub(r"\n```\s*$", "", text)

    # Match one or more Aider-style blocks.  The path is the first line of the
    # block, followed by the delimited SEARCH/REPLACE sections.
    pattern = re.compile(
        r"^(?P<path>[^\n]+)\n<<<<<<< SEARCH\n(?P<search>.*?)\n=======\n(?P<replace>.*?)\n>>>>>>> REPLACE",
        re.DOTALL | re.MULTILINE,
    )
    blocks = []
    for match in pattern.finditer(text):
        blocks.append((match.group("path").strip(), match.group("search"), match.group("replace")))
    return blocks


def compute_format_reward(patch_text: str, task_dir: Path | str) -> float:
    """Compute a cheap, graduated format-shaping reward for a generated patch.

    This reward does not execute the task.  It checks for well-formed
    SEARCH/REPLACE blocks and, for each block, uses `difflib` to measure how
    much of the SEARCH text is present in the claimed target file.  The fuzzy
    score gives the model a gradient toward correct line-matching without
    requiring an exact match from the start.
    """
    from ..task import load_task

    task_dir = Path(task_dir)
    task = load_task(task_dir)
    workspace = task_dir / "repo"
    allowed = set(task.allowed_paths)

    blocks = _aider_blocks(patch_text)
    if not blocks:
        return 0.0

    reward = 0.0
    for path, search, _ in blocks:
        reward += FORMAT_BLOCK_REWARD

        if not (workspace / path).exists():
            continue
        if path not in allowed:
            continue

        if not search:
            continue

        file_text = (workspace / path).read_text(encoding="utf-8")
        # Sum the lengths of all exact matching blocks between search and file.
        matcher = difflib.SequenceMatcher(None, search, file_text)
        matching = sum(block.size for block in matcher.get_matching_blocks())
        similarity = min(matching / len(search), 1.0)
        reward += FORMAT_FUZZY_SCALE * similarity

    return reward
