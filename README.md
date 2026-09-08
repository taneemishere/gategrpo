# PatchProof

**Gate-Verified Repair Search and Reproducible Benchmarks for Coding Agents.**

PatchProof is the reliability layer around a coding agent's candidate patches: it verifies every candidate through hard gates, turns failures into bounded evidence, routes the next attempt deliberately, and leaves an auditable trail of why each patch was rejected or promoted. It is not a coding agent itself — it is the controller that makes a repair loop trustworthy.

Read the full write-up here: **https://taneemishere.github.io/patchproof/**

## What it does

Every candidate patch flows through the same pipeline:

```text
candidate patch
  -> hard gates        (scope guard, patch applies, AST parse, secret scan, visible tests, hidden release-gate)
  -> bounded evidence  (a compact failure packet — never the raw logs — so the next attempt is targeted, not leaky)
  -> route-aware next  (the critic maps the failure to a repair route and picks a compatible next candidate)
  -> archive + lineage (every attempt is recorded with parent, route, fingerprint, and outcome)
  -> narrative + benchmark reports
```

Features you can drive from the CLI (each has a command below):

- **Hard verification gates** — a patch is only promoted after it passes scope, syntax, secret, visible-test, and hidden regression checks.
- **Bounded evidence** — visible failures expose only trimmed output; hidden release-gate failures expose failing test *names* but withhold values, so the loop can't overfit to hidden tests.
- **Route-aware selection** — failures are routed to a repair strategy and matched to candidates by declared metadata, not filename guessing.
- **Repeated-failure diversification** — when the same fix keeps failing, the controller pivots instead of resampling the same dead end.
- **Auditable artifacts** — candidate archive, lineage, failure fingerprints, and a one-page narrative report per run.
- **Runs offline or live** — deterministic curated patches (no API key) *or* a live LLM generating patches, through the exact same gates.
- **Single-file or multi-file repairs** — from a one-function fix to a coordinated change across several modules.
- **Bring your own code** — scaffold any buggy repo into a task and repair it.

## How to run

Install dependencies:

```bash
python3 -m pip install -r requirements-dev.txt
```

### Command map

Use the commands below according to the job at hand. The first five are the normal product workflows; the last four are useful when inspecting, reproducing, or researching a repair run.

| Command | When to use it |
| --- | --- |
| `demo` | Run the complete local demonstration: hero repair, benchmark comparison, policy experiment, and reports. |
| `showcase` | Repair one task and create the focused one-page HTML report. This is the best live demonstration command. |
| `init-task` | Turn your own buggy repository into a PatchProof task, or validate an existing task. |
| `benchmark` | Compare deterministic curated candidates across the repair-controller baselines. |
| `benchmark-llm` | Benchmark a live LLM across the suite using the same gates and search controller. |
| `run` | Verify one supplied patch through the deterministic hard gates. |
| `search` | Run deterministic repair search over a task's curated patches or explicitly supplied patch files. |
| `report` | Render a static HTML report from an existing run directory. |
| `policy-experiment` | Tune and evaluate repair-controller policies from an experiment configuration. |

### The full local demo

Runs a hero repair + the benchmark across baselines + a policy experiment, and writes HTML reports.

Offline (deterministic, no API key):

```bash
python3 -m patchproof demo --run-dir .patchproof_runs/demo
```

Live LLM (the model generates the hero repair candidates, then the same gates run). Point `--llm-base-url` at your OpenAI-compatible vLLM server:

```bash
python3 -m patchproof demo --run-dir .patchproof_runs/demo --llm \
  --llm-provider vllm --llm-base-url http://localhost:8000/v1
```

Both write `hero_report.html`, `benchmark_report.html`, and `demo_summary.json` under the run directory.

### Showcase a single repair

`showcase` runs one repair on a single task and renders a narrative, one-page HTML report (Problem → Attempt Timeline → Evidence & Routing → Gates → Promoted Patch). Use it to *see* how a fix gets verified, routed, and promoted — this is the best command for a live demo.

```bash
# offline (deterministic, uses the task's curated patches):
python3 -m patchproof showcase tasks/string_slugifier

# live (a model generates the patches):
python3 -m patchproof showcase tasks/string_slugifier --llm \
  --llm-provider vllm --llm-base-url http://localhost:8000/v1
```

The report is written to `.patchproof_runs/showcase/showcase_report.html`.

### Multi-file repair

`tasks/order_service` is a coordinated **three-file** repair: the fix must span `shop/pricing.py`, `shop/inventory.py`, and `shop/checkout.py` at once (rounding, validation, and checkout flow) to satisfy the spec, while leaving the catalog and tests untouched. The scope guard blocks any patch that strays outside those three files.

```bash
# offline:
python3 -m patchproof showcase tasks/order_service

# live:
python3 -m patchproof showcase tasks/order_service --llm \
  --llm-provider vllm --llm-base-url http://localhost:8000/v1
```

### Bring your own code

`init-task` turns any buggy repository into a self-contained PatchProof task. The repo ships a runnable single-file example at `examples/bring_your_own_code/`:

```text
examples/bring_your_own_code/
├── instructions.md
├── src/
│   └── stats.py
└── tests/
    ├── visible/
    │   └── test_average.py
    └── regression/
        └── test_average_empty.py
```

`instructions.md` (the spec the repair must satisfy):

```markdown
# Repair `average`

Fix `average` in `stats.py` so it returns the arithmetic mean of the numbers.
For an empty list it must return `0.0` instead of raising.
```

`src/stats.py` (intentionally buggy — crashes on an empty list):

```python
def average(numbers):
    return sum(numbers) / len(numbers)
```

Scaffold it into a task, then repair it:

```bash
python3 -m patchproof init-task ./my_task \
  --repo examples/bring_your_own_code/src \
  --instructions examples/bring_your_own_code/instructions.md \
  --allowed-path stats.py \
  --visible-tests examples/bring_your_own_code/tests/visible \
  --regression-tests examples/bring_your_own_code/tests/regression

python3 -m patchproof showcase ./my_task --llm \
  --llm-provider vllm --llm-base-url http://localhost:8000/v1
```

`init-task` copies the source, tests, and instructions into `./my_task` and writes a valid `task.json`. A fresh task has no curated patches, so the live `--llm` path is what generates the fix. Omit the test flags to scaffold placeholder tests so the task runs immediately.

**Multi-file?** Pass `--repo` a package directory and repeat `--allowed-path` for each file the fix is allowed to touch:

```bash
python3 -m patchproof init-task ./my_task \
  --repo ./my_pkg \
  --instructions ./instructions.md \
  --allowed-path shop/pricing.py \
  --allowed-path shop/checkout.py \
  --visible-tests ./tests/visible \
  --regression-tests ./tests/regression
```

Validate an existing task without changing anything:

```bash
python3 -m patchproof init-task ./my_task --validate
```

### Benchmarks

For reproducible, apples-to-apples numbers across baselines (single-shot, linear retry, evidence-aware, full PatchProof, …):

```bash
# deterministic, offline: replays curated patches across the baselines
python3 -m patchproof benchmark benchmarks/repair_routing_suite.json --run-dir .patchproof_runs/benchmark

# live: the model generates patches per task family, then the same gates/search run
python3 -m patchproof benchmark-llm benchmarks/repair_routing_suite.json \
  --llm-provider vllm --llm-base-url http://localhost:8000/v1
```

The live benchmark summary also reports token and wall-clock cost per task, so you can weigh reliability against cost.

### Tests

```bash
python3 -m pytest -q tests
```
