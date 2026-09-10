"""SFT warm-up for GateGRPO-RL.

Trains a LoRA adapter on the existing candidate patch files, merges it into the
base model, and saves the merged checkpoint.  Candidate patches are stored as
unified diffs; this script converts them into the Aider SEARCH/REPLACE format
that the prompt and the verifier expect.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import uuid
from io import StringIO
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from unidiff import PatchSet

from ..llm import _initial_prompt
from ..runner import run_task


def _patch_applies(task_dir: Path, patch_text: str) -> bool:
    """Run the verifier and return whether the patch_applies gate passes."""
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".patch", delete=False
    ) as f:
        f.write(patch_text)
        patch_path = Path(f.name)

    run_root = Path(".gategrpo_runs/sft_verify")
    run_root.mkdir(parents=True, exist_ok=True)
    run_dir = run_root / f"{task_dir.name}_{uuid.uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = run_task(task_dir, run_dir, patch_path)
        return result.status == "promoted"
    finally:
        patch_path.unlink(missing_ok=True)
        shutil.rmtree(run_dir, ignore_errors=True)
    return False


def _unified_to_aider(patch_text: str) -> str:
    """Convert a git unified diff into Aider SEARCH/REPLACE blocks.

    Each hunk becomes one Aider block.  The SEARCH text is the original
    source (context + removed lines) and the REPLACE text is the new source
    (context + added lines).  Paths are reported as in the git diff, with the
    leading `b/` stripped.
    """
    try:
        patchset = PatchSet(StringIO(patch_text))
    except Exception:
        return ""

    blocks: list[str] = []
    for patch in patchset:
        target = patch.target_file
        if not target or target == "/dev/null":
            continue
        if target.startswith("b/"):
            target = target[2:]

        for hunk in patch:
            search_lines = [line[1:] for line in hunk.source]
            replace_lines = [line[1:] for line in hunk.target]
            search = "".join(search_lines).rstrip("\n")
            replace = "".join(replace_lines).rstrip("\n")
            block = (
                f"{target}\n"
                "<<<<<<< SEARCH\n"
                f"{search}\n"
                "=======\n"
                f"{replace}\n"
                ">>>>>>> REPLACE\n"
            )
            blocks.append(block)

    return "\n".join(blocks) if blocks else ""


def build_sft_dataset(
    tasks_root: Path,
    tokenizer: AutoTokenizer,
    max_prompt_length: int,
) -> Dataset:
    """Build an SFT dataset from candidate patch files that pass patch_applies."""
    rows: list[dict[str, str]] = []
    for task_dir in sorted(tasks_root.glob("*")):
        if not task_dir.is_dir():
            continue

        prompt_text = _initial_prompt(task_dir)
        tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
        if len(tokens) > max_prompt_length - 50:
            tokens = tokens[: max_prompt_length - 50]
            prompt_text = tokenizer.decode(tokens, skip_special_tokens=True)

        for patch_file in sorted(task_dir.glob("candidate*.patch")):
            patch_text = patch_file.read_text(encoding="utf-8")
            if not _patch_applies(task_dir, patch_text):
                continue
            aider_text = _unified_to_aider(patch_text)
            if not aider_text:
                continue
            messages = [
                {"role": "user", "content": prompt_text},
                {"role": "assistant", "content": aider_text},
            ]
            rows.append({"messages": messages})

    return Dataset.from_list(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gategrpo-rl-sft-warmup",
        description="SFT warm-up on existing repair patches.",
    )
    parser.add_argument(
        "--tasks-root",
        type=Path,
        default=Path("tasks"),
        help="Directory containing task folders with .patch files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/gategrpo-sft-1.5b"),
        help="Directory to write the LoRA adapter and merged model.",
    )
    parser.add_argument(
        "--merged-subdir",
        type=str,
        default="merged",
        help="Subdirectory under output-dir for the merged full model.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        help="HuggingFace model id or local path.",
    )
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=8192,
        help="Max prompt tokens (truncation budget before the patch).",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=32768,
        help="Max total sequence tokens (prompt + patch + template).",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=int,
        default=10,
        help="Number of SFT epochs over the candidate patch corpus.",
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

    if not args.tasks_root.exists():
        print(f"Tasks root not found: {args.tasks_root}")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = build_sft_dataset(args.tasks_root, tokenizer, args.max_prompt_length)
    print(f"Built SFT dataset with {len(dataset)} examples from {args.tasks_root}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        device_map="auto",
        trust_remote_code=True,
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

    sft_args = SFTConfig(
        output_dir=str(args.output_dir / "adapter"),
        num_train_epochs=args.num_train_epochs,
        learning_rate=5e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_length=args.max_seq_length,
        dataset_text_field="messages",
        bf16=args.bf16,
        completion_only_loss=True,
        logging_steps=1,
        save_steps=50,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        peft_config=peft_config,
        args=sft_args,
    )

    trainer.train()
    trainer.save_model(str(args.output_dir / "adapter"))

    # Merge the adapter into the base model so the GRPO trainer can load it as
    # a normal causal-LM checkpoint.
    print("Merging LoRA adapter into the base model...")
    merged = trainer.model.merge_and_unload()
    merged_dir = args.output_dir / args.merged_subdir
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))
    print(f"SFT warm-up complete. Merged model saved to {merged_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
