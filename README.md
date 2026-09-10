# GateGRPO

A gate-verified repair-search and RL training harness for code repair. GateGRPO runs a deterministic hard-gate verifier on every candidate patch, turns failures into bounded evidence, and uses the verifier's per-gate outputs as a reward source for reinforcement learning.

**Read the full write-up:** https://taneemishere.github.io/gategrpo

## Install

```bash
python3 -m pip install -r requirements.txt
```

## Quick start

Run the local demo (offline, deterministic):

```bash
python3 -m gategrpo demo --run-dir .gategrpo_runs/demo
```

Run the demo with a live model served by vLLM:

```bash
python3 -m gategrpo demo --run-dir .gategrpo_runs/demo --llm \
  --llm-provider vllm --llm-base-url http://localhost:8000/v1
```

Repair a single task and see a narrative HTML report:

```bash
python3 -m gategrpo showcase tasks/string_slugifier
```

Scaffold your own repository as a task:

```bash
python3 -m gategrpo init-task ./my_task \
  --repo examples/bring_your_own_code/src \
  --instructions examples/bring_your_own_code/instructions.md \
  --allowed-path stats.py \
  --visible-tests examples/bring_your_own_code/tests/visible \
  --regression-tests examples/bring_your_own_code/tests/regression
```

## Training and evaluation

The training scripts live in `gategrpo/rl/`:

```bash
python3 -m gategrpo.rl.baseline          # evaluate a model on the RL pilot suite
python3 -m gategrpo.rl.sft_warmup        # run SFT warm-up
python3 -m gategrpo.rl.grpo_trainer      # run the GRPO training loop
python3 -m gategrpo.rl.inspect_patches   # inspect generated patches
```

Run the deterministic benchmark:

```bash
python3 -m gategrpo benchmark benchmarks/repair_routing_suite.json --run-dir .gategrpo_runs/benchmark
```

Run the LLM benchmark:

```bash
python3 -m gategrpo benchmark-llm benchmarks/repair_routing_suite.json \
  --llm-provider vllm --llm-base-url http://localhost:8000/v1
```

Run tests:

```bash
python3 -m pytest -q tests
```

## Code layout

- `gategrpo/` — the verification harness, gates, and search controller
- `gategrpo/rl/` — SFT, GRPO, and reward code
- `benchmarks/` — task suites and held-out evaluations
- `tasks/` — benchmark tasks used by the suites
- `examples/` — bring-your-own-code example
- `tests/` — unit and integration tests
- `scripts/` — helper scripts for training and inference
