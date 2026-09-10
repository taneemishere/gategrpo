from __future__ import annotations

import json
import shutil
from pathlib import Path

from .llm import run_llm_search
from .models import CandidateSpec, SearchBudget


def generate_candidate_pool(
    task_dir: Path,
    run_dir: Path,
    *,
    client,
    budget: SearchBudget,
    provider: str = "openrouter",
    temperature: float | None = None,
    seed: int | None = None,
) -> list[CandidateSpec]:
    generation_dir = run_dir / "generated_pool"
    generation_dir.mkdir(parents=True, exist_ok=True)
    result = run_llm_search(
        task_dir,
        run_dir / "llm_generation",
        budget=budget,
        client=client,
        provider=provider,
        temperature=temperature,
        seed=seed,
    )

    pool: list[CandidateSpec] = []
    artifact_rows: list[dict[str, object]] = []
    for record in result.records:
        patch_path = Path(record.patch_file)
        if not patch_path.exists() or patch_path.stat().st_size == 0:
            continue
        copied_patch = generation_dir / f"{record.candidate_id}.patch"
        shutil.copyfile(patch_path, copied_patch)
        # Prefer metadata the generator emitted about its own candidate; fall
        # back to controller-derived signals when the model declared none.
        generator_declared = record.metadata_source == "generator"
        intent = record.candidate_intent or (record.failure_type or "promoted")
        compatible_routes = record.compatible_routes or (record.route,)
        spec = CandidateSpec(
            path=copied_patch,
            label=record.candidate_id,
            generator="llm_live_declared" if generator_declared else "llm_live",
            intent=intent,
            compatible_routes=compatible_routes,
        )
        pool.append(spec)
        artifact_rows.append(
            {
                "label": spec.label,
                "generator": spec.generator,
                "metadata_source": record.metadata_source,
                "intent": spec.intent,
                "compatible_routes": list(spec.compatible_routes),
                "path": str(spec.path),
            }
        )

    artifact = {
        "task_id": result.task_id,
        "suite_run_dir": str(run_dir),
        "generated_pool": artifact_rows,
    }
    (generation_dir / "generated_pool.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return pool
