"""GRPO training driver for GateGRPO-RL.

Uses TRL's GRPOTrainer with a custom reward function that runs every generated
candidate through the GateGRPO hard-gate verifier.  No learned reward model is
needed; the reward is the verifier itself.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from ..llm import _initial_prompt
from ..runner import run_task
from .rewards import compute_format_reward, gategrpo_reward


def _temp_run_dir(task_id: str, run_root: Path) -> Path:
    """Create a unique run directory for one generated candidate."""
    run_root.mkdir(parents=True, exist_ok=True)
    # uuid keeps parallel GRPO samples from colliding
    run_dir = run_root / f"{task_id.replace('/', '_')}_{uuid.uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def gategrpo_reward_func(
    prompts: list[list[dict[str, Any]]],
    completions: list[list[dict[str, Any]]],
    task_dir: list[str] | None = None,
    **kwargs: Any,
) -> list[float]:
    """TRL GRPO reward function backed by the GateGRPO verifier.

    Args:
        prompts: Batch of prompt messages (one per generated completion).
        completions: Batch of generated completion messages.
        task_dir: Dataset column with the task directory for each completion.
        **kwargs: Extra dataset columns passed by TRL.

    Returns:
        A list of scalar rewards, one per completion.
    """
    if task_dir is None:
        raise ValueError("gategrpo_reward_func requires a 'task_dir' column in the dataset")

    run_root = Path(".gategrpo_runs/grpo_rollouts")
    rewards: list[float] = []

    for prompt, completion, tdir in zip(prompts, completions, task_dir):
        # completion is a list of message dicts; the assistant turn is the first.
        patch_text = completion[0]["content"]

        # Write the generated patch to a temp file
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".patch", delete=False
        ) as f:
            f.write(patch_text)
            patch_file = Path(f.name)

        try:
            task_path = Path(tdir)
            # unique run dir for this candidate
            run_dir = _temp_run_dir(task_path.name, run_root)
            result = run_task(task_path, run_dir, patch_file)
            # Reward ignores token/cost; those are tracked by GateGRPO separately.
            reward = gategrpo_reward(
                gates=result.gates,
                promoted=result.passed,
                completion_tokens=0,
                max_tokens=1,
            )
        finally:
            patch_file.unlink(missing_ok=True)

        rewards.append(reward)

    return rewards


def format_reward_func(
    prompts: list[list[dict[str, Any]]],
    completions: list[list[dict[str, Any]]],
    task_dir: list[str] | None = None,
    **kwargs: Any,
) -> list[float]:
    """Cheap format-shaping reward for Aider-style SEARCH/REPLACE blocks."""
    if task_dir is None:
        raise ValueError("format_reward_func requires a 'task_dir' column in the dataset")

    rewards: list[float] = []
    for completion, tdir in zip(completions, task_dir):
        patch_text = completion[0]["content"]
        rewards.append(compute_format_reward(patch_text, tdir))
    return rewards


def build_prompt_dataset(
    suite_path: Path,
    tokenizer: AutoTokenizer,
    max_prompt_length: int,
) -> Dataset:
    """Build a TRL dataset where each row is one task's initial prompt."""
    with suite_path.open(encoding="utf-8") as f:
        suite = json.load(f)

    # Reserve tokens for the chat template wrappers.
    token_budget = max_prompt_length - 50

    rows = []
    for case in suite["cases"]:
        task_dir = Path(case["task_dir"])
        prompt_text = _initial_prompt(task_dir)

        # Truncate to the token budget so the prompt fits under max_prompt_length.
        tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
        if len(tokens) > token_budget:
            tokens = tokens[:token_budget]
            prompt_text = tokenizer.decode(tokens, skip_special_tokens=True)

        rows.append(
            {
                "prompt": [{"role": "user", "content": prompt_text}],
                "task_dir": str(task_dir),
                "task_id": case.get("id", task_dir.name),
            }
        )
    return Dataset.from_list(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gategrpo-rl-train-grpo",
        description="Train a GateGRPO repair policy with GRPO.",
    )
    parser.add_argument(
        "--suite",
        type=Path,
        default=Path("benchmarks/rl_pilot_suite.json"),
        help="Path to the benchmark suite JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/gategrpo-grpo-1.5b-v3"),
        help="Directory to write the final checkpoint.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        help="HuggingFace model id or local path (use the merged SFT checkpoint if available).",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=8,
        help="Number of completions to sample per prompt (GRPO group size).",
    )
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=8192,
        help="Max prompt tokens (must fit in the model's context).",
    )
    parser.add_argument(
        "--max-completion-length",
        type=int,
        default=512,
        help="Max completion tokens per candidate.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-6,
        help="GRPO learning rate.",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=int,
        default=3,
        help="Number of epochs over the pilot suite.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.05,
        help="KL divergence coefficient (higher = stronger anchoring to SFT init).",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=50,
        help="Save a checkpoint every N steps for intermediate evaluation.",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank.",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha.",
    )
    parser.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use bfloat16 training.",
    )
    args = parser.parse_args(argv)

    if not args.suite.exists():
        print(f"Suite not found: {args.suite}")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(args.output_dir, ignore_errors=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Remove any old GRPO rollout workspaces so we start clean.
    rollout_root = Path(".gategrpo_runs/grpo_rollouts")
    shutil.rmtree(rollout_root, ignore_errors=True)

    # Qwen2.5-Coder uses an explicit pad token.
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = build_prompt_dataset(args.suite, tokenizer, args.max_prompt_length)
    print(f"Loaded {len(dataset)} prompts from {args.suite}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        device_map="auto",
        trust_remote_code=True,
        # SDPA is the safe default that works without flash-attn installed.
        attn_implementation="sdpa",
    )

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    grpo_args = GRPOConfig(
        output_dir=str(args.output_dir),
        learning_rate=args.learning_rate,
        num_generations=args.num_generations,
        generation_batch_size=args.num_generations,
        max_completion_length=args.max_completion_length,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=args.num_train_epochs,
        bf16=args.bf16,
        # Expert recommendation: Dr. GRPO avoids killing gradients when a group's
        # reward std is zero, and batch reward scaling avoids zero-std collapse.
        loss_type="dr_grpo",
        scale_rewards="batch",
        beta=args.beta,
        # do not use an external vLLM server; HF generation keeps the single-GPU
        # setup simple and the verifier is the bottleneck anyway.
        use_vllm=False,
        logging_steps=1,
        save_steps=args.save_steps,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[gategrpo_reward_func, format_reward_func],
        args=grpo_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(str(args.output_dir))
    print(f"GRPO training complete. Model saved to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
