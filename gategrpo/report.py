from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any


def generate_html_report(input_dir: Path, output_path: Path) -> Path:
    sections: list[str] = []
    search_archive = input_dir / "candidate_archive.json"
    benchmark_results = input_dir / "benchmark_results.json"

    if search_archive.exists():
        sections.append(_search_section(search_archive))
    if benchmark_results.exists():
        sections.append(_benchmark_section(benchmark_results))
        sections.extend(_nested_search_sections(input_dir, benchmark_results))

    if not sections:
        raise ValueError(f"no GateGRPO report artifacts found in {input_dir}")

    html = _page("GateGRPO Report", "\n".join(sections))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def generate_model_matrix_report(input_dir: Path, output_path: Path) -> Path:
    """Render the model-matrix experiment as a single comparative HTML page.

    Reads ``model_matrix_results.json`` (written by ``run_llm_model_matrix``) from
    ``input_dir`` and produces a comparison of every model family: an overall
    table plus a per-task solve@budget matrix so models are compared side by side
    rather than across separate JSON/Markdown artifacts.
    """
    results_path = input_dir / "model_matrix_results.json"
    if not results_path.exists():
        raise ValueError(f"no GateGRPO model matrix artifact found in {input_dir}")
    payload = _read_json(results_path)
    matrix = payload.get("matrix", [])
    if not isinstance(matrix, list):
        matrix = []
    suite = str(payload.get("suite", "unknown"))
    sections = [
        _matrix_overall_section(suite, matrix, results_path),
        _matrix_per_task_section(matrix),
    ]
    html = _page("GateGRPO Model Matrix", "\n".join(sections))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def _format_summary_cost(summary: dict[str, Any]) -> str:
    avg_cost = summary.get("avg_cost")
    if avg_cost is None:
        return "n/a"
    return f"{avg_cost} {summary.get('currency', 'USD')}"


def _pm(value: object, stdev: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(stdev, (int, float)) and stdev:
        return f"{value} ± {stdev}"
    return str(value)


def _matrix_overall_section(suite: str, matrix: list[dict[str, Any]], results_path: Path) -> str:
    rows = "\n".join(
        "<tr>"
        f"<td>{escape(str(entry.get('id', entry.get('model', 'model'))))}</td>"
        f"<td>{escape(str(entry.get('model', 'unknown')))}</td>"
        f"<td>{escape(str(entry.get('provider', 'unknown')))}</td>"
        f"<td>{escape(str(entry.get('overall', {}).get('runs', 0)))}</td>"
        f"<td>{escape(str(entry.get('overall', {}).get('solve_at_budget', 0.0)))}</td>"
        f"<td>{escape(_ci_text(entry.get('overall', {}).get('solve_at_budget_ci95')))}</td>"
        f"<td>{escape(str(entry.get('overall', {}).get('avg_attempts', 0.0)))}</td>"
        f"<td>{escape(_pm(entry.get('overall', {}).get('avg_tokens', 0.0), entry.get('overall', {}).get('tokens_stdev')))}</td>"
        f"<td>{escape(str(entry.get('overall', {}).get('token_source', 'estimate')))}</td>"
        f"<td>{escape(_format_summary_cost(entry.get('overall', {})))}</td>"
        f"<td>{escape(_pm(entry.get('overall', {}).get('avg_wall_seconds', 0.0), entry.get('overall', {}).get('wall_seconds_stdev')))}</td>"
        "</tr>"
        for entry in matrix
    )
    if not rows:
        rows = "<tr><td colspan='11' class='muted'>No models in matrix.</td></tr>"
    return f"""
<section>
  <h2>Model comparison: {escape(suite)}</h2>
  <p class="muted">Results: {escape(str(results_path))}</p>
  <table>
    <thead>
      <tr>
        <th>ID</th>
        <th>Model</th>
        <th>Provider</th>
        <th>Runs</th>
        <th>solve@budget</th>
        <th>95% CI</th>
        <th>Avg attempts</th>
        <th>Avg tokens (± stdev)</th>
        <th>Token source</th>
        <th>Avg cost</th>
        <th>Avg wall s (± stdev)</th>
      </tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
</section>
"""


def _matrix_per_task_section(matrix: list[dict[str, Any]]) -> str:
    task_ids: list[str] = []
    for entry in matrix:
        per_task = entry.get("per_task", {})
        if not isinstance(per_task, dict):
            continue
        for task_id in per_task:
            if task_id not in task_ids:
                task_ids.append(task_id)
    task_ids.sort()
    if not task_ids or not matrix:
        return """
<section>
  <h2>Per-task solve@budget</h2>
  <p class="muted">No per-task results available.</p>
</section>
"""
    headers = "".join(f"<th>{escape(str(entry.get('id', entry.get('model', 'model'))))}</th>" for entry in matrix)
    body_rows = []
    for task_id in task_ids:
        scores = [
            entry.get("per_task", {}).get(task_id, {}).get("solve_at_budget")
            if isinstance(entry.get("per_task", {}), dict)
            else None
            for entry in matrix
        ]
        numeric = [score for score in scores if isinstance(score, (int, float))]
        best = max(numeric) if numeric else None
        cells = []
        for entry, score in zip(matrix, scores):
            if score is None:
                cells.append("<td class='muted'>n/a</td>")
                continue
            ci = entry.get("per_task", {}).get(task_id, {}).get("solve_at_budget_ci95")
            css = " class='best'" if best is not None and score == best else ""
            cells.append(f"<td{css}>{escape(str(score))}<br><span class='muted'>{escape(_ci_text(ci))}</span></td>")
        body_rows.append(f"<tr><td>{escape(task_id)}</td>{''.join(cells)}</tr>")
    return f"""
<section>
  <h2>Per-task solve@budget</h2>
  <p class="muted">Best model per task highlighted. Cells show solve@budget with 95% CI.</p>
  <table>
    <thead>
      <tr><th>Task</th>{headers}</tr>
    </thead>
    <tbody>{''.join(body_rows)}</tbody>
  </table>
</section>
"""


def generate_showcase_report(input_dir: Path, output_path: Path) -> Path:
    archive_path = input_dir / "candidate_archive.json"
    if not archive_path.exists():
        raise ValueError(f"no GateGRPO showcase archive found in {input_dir}")
    archive = _read_json(archive_path)
    run = archive.get("run", {})
    if not isinstance(run, dict):
        run = {}
    records = archive.get("records", [])
    if not isinstance(records, list):
        records = []
    sections = [
        _showcase_run_summary_section(run, archive_path, len(records)),
        _showcase_problem_section(run),
        _showcase_attempt_timeline_section(records),
        _showcase_evidence_and_routing_section(records),
        _showcase_gate_results_section(records),
        _showcase_promoted_patch_section(records),
        _showcase_artifacts_section(input_dir, run),
    ]
    html = _page("GateGRPO Showcase", "\n".join(sections))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def _showcase_candidate_card(record: dict[str, Any], index: int) -> str:
    status = str(record.get("status", "unknown"))
    return f"""
<article class="candidate {escape(status)}">
  <h3>{index}. {escape(str(record.get('candidate_id', 'candidate')))} — {escape(status)}</h3>
  <dl>
    <dt>Route</dt><dd>{escape(str(record.get('route', 'unknown')))}</dd>
    <dt>Selected by</dt><dd>{escape(str(record.get('selection_reason') or 'unknown'))}</dd>
    <dt>Intent</dt><dd>{escape(str(record.get('candidate_intent') or 'unknown'))}</dd>
    <dt>Gen</dt><dd>{escape(str(record.get('generation', 0)))}</dd>
    <dt>Failure</dt><dd>{escape(str(record.get('failure_type') or 'none'))}</dd>
    <dt>Fingerprint</dt><dd>{escape(str(record.get('failure_fingerprint') or 'none'))}</dd>
  </dl>
</article>
"""


def _dl(items: list[tuple[str, object]]) -> str:
    return "<dl>" + "".join(
        f"<dt>{escape(str(label))}</dt><dd>{escape(str(value))}</dd>" for label, value in items
    ) + "</dl>"


def _format_run_cost(run: dict[str, Any]) -> str:
    cost = run.get("cost")
    if cost is None:
        return "n/a"
    return f"{cost} {run.get('currency', 'USD')}"


def _format_budget(budget: object) -> str:
    if not isinstance(budget, dict) or not budget:
        return "unknown"
    parts = [f"{key}={budget[key]}" for key in ("max_candidates", "max_repeated_failures") if key in budget]
    extras = [f"{key}={value}" for key, value in budget.items() if key not in {"max_candidates", "max_repeated_failures"}]
    parts.extend(extras)
    return ", ".join(parts) if parts else "unknown"


def _format_list(values: object) -> str:
    if isinstance(values, (list, tuple)):
        return ", ".join(str(value) for value in values) or "none"
    if values is None:
        return "none"
    return str(values)


def _showcase_run_summary_section(run: dict[str, Any], archive_path: Path, attempts: int) -> str:
    budget = run.get("budget", {}) if isinstance(run, dict) else {}
    items = [
        ("Status", run.get("status") or "unknown"),
        ("Stop reason", run.get("stop_reason") or "unknown"),
        ("Attempts", run.get("attempts", attempts)),
        ("Model", run.get("model") or "unknown"),
        ("Provider", run.get("provider") or "unknown"),
        ("Budget", _format_budget(budget)),
        ("Total tokens", run.get("total_tokens", "unknown")),
        ("Token source", run.get("token_source", "estimate")),
        ("Prompt tokens", run.get("prompt_tokens", "unknown")),
        ("Completion tokens", run.get("completion_tokens", "unknown")),
        ("Cost", _format_run_cost(run)),
        ("Wall seconds", run.get("wall_seconds", "unknown")),
        ("Repair path", run.get("repair_path") or "unknown"),
    ]
    return f"""
<section>
  <h2>Run Summary</h2>
  <p class="muted">Archive: {escape(str(archive_path))}</p>
  {_dl(items)}
</section>
"""


def _showcase_problem_section(run: dict[str, Any]) -> str:
    items = [
        ("Task", run.get("task_name", run.get("task_id", "unknown"))),
        ("Allowed paths", ", ".join(run.get("allowed_paths", [])) or "unknown"),
    ]
    instructions = escape(str(run.get("instructions_excerpt", "No instructions excerpt available.")))
    return f"""
<section>
  <h2>Problem</h2>
  {_dl(items)}
  <h3>Instructions excerpt</h3>
  <pre class="excerpt">{instructions}</pre>
</section>
"""


def _showcase_attempt_timeline_section(records: list[dict[str, Any]]) -> str:
    cards = "\n".join(_showcase_candidate_card(record, index) for index, record in enumerate(records, start=1))
    return f"""
<section>
  <h2>Attempt Timeline</h2>
  <div class="cards timeline">{cards}</div>
</section>
"""


def _showcase_evidence_and_routing_section(records: list[dict[str, Any]]) -> str:
    cards = []
    for index, record in enumerate(records):
        evidence = record.get("evidence")
        if not evidence:
            continue
        next_route = records[index + 1].get("route", "stop") if index + 1 < len(records) else "stop"
        details = json.dumps(evidence.get("details", {}), indent=2, sort_keys=True)
        cards.append(
            f"""
<article class="candidate evidence-card">
  <h3>{escape(str(record.get('candidate_id', 'candidate')))} — {escape(str(record.get('route', 'unknown')))}</h3>
  <p><strong>Evidence:</strong> {escape(str(evidence.get('summary', 'failure evidence')))}</p>
  <p class="muted"><strong>Next route:</strong> {escape(str(next_route))}</p>
  <pre>{escape(details)}</pre>
</article>
"""
        )
    if not cards:
        cards.append("<p class='muted'>No failure evidence was recorded.</p>")
    return f"""
<section>
  <h2>Evidence &amp; Routing</h2>
  <div class="cards">{''.join(cards)}</div>
</section>
"""


def _showcase_gate_results_section(records: list[dict[str, Any]]) -> str:
    cards = []
    for record in records:
        gate_items = "\n".join(
            f"<li class='{_status_class(bool(gate.get('passed')))}'>{escape(str(gate.get('name', 'gate')))}: {'PASS' if gate.get('passed') else 'FAIL'}</li>"
            for gate in record.get("gates", [])
        )
        cards.append(
            f"""
<article class="candidate gate-card">
  <h3>{escape(str(record.get('candidate_id', 'candidate')))}</h3>
  <ul class="gates">{gate_items}</ul>
</article>
"""
        )
    return f"""
<section>
  <h2>Gate Results</h2>
  <div class="cards">{''.join(cards)}</div>
</section>
"""


def _showcase_promoted_patch_section(records: list[dict[str, Any]]) -> str:
    promoted = next((record for record in records if record.get("status") == "promoted"), None)
    if promoted is None:
        last = records[-1] if records else {}
        patch_html = _patch_snippet(last.get("patch_file"))
        touched_files = _format_list(last.get("touched_files", []))
        return f"""
<section>
  <h2>Promoted Patch</h2>
  <p class="muted">No candidate was promoted; showing the last attempt instead.</p>
  <p><strong>Last attempt:</strong> {escape(str(last.get('candidate_id', 'none')))} — {escape(str(last.get('status', 'unknown')))}</p>
  <p><strong>Touched files:</strong> {escape(touched_files)}</p>
  {patch_html}
</section>
"""
    patch_html = _patch_snippet(promoted.get("patch_file"))
    touched_files = _format_list(promoted.get("touched_files", []))
    return f"""
<section>
  <h2>Promoted Patch</h2>
  <p><strong>Candidate:</strong> {escape(str(promoted.get('candidate_id', 'candidate')))}</p>
  <p><strong>Touched files:</strong> {escape(touched_files)}</p>
  {patch_html}
</section>
"""


def _showcase_artifacts_section(input_dir: Path, run: dict[str, Any]) -> str:
    llm_files = sorted((input_dir / "llm").glob("*_response.json")) if (input_dir / "llm").exists() else []
    artifact_items = [
        f"candidate_archive.json → {escape(str(input_dir / 'candidate_archive.json'))}",
        f"controller_trace.jsonl → {escape(str(input_dir / 'controller_trace.jsonl'))}",
        f"workspace → {escape(str(run.get('workspace', 'unknown')))}",
    ]
    if llm_files:
        artifact_items.extend(f"LLM response → {escape(str(path))}" for path in llm_files)
    else:
        artifact_items.append("LLM response files → none present")
    return f"""
<section>
  <h2>Artifacts</h2>
  <ul class="artifact-list">
    {''.join(f'<li>{item}</li>' for item in artifact_items)}
  </ul>
</section>
"""


def _search_section(archive_path: Path, title: str = "Search trace") -> str:
    archive = _read_json(archive_path)
    records = archive.get("records", [])
    cards = "\n".join(_candidate_card(record, index) for index, record in enumerate(records, start=1))
    lineage = " → ".join(
        f"<span class='node {escape(record.get('status', ''))}'>{escape(record.get('candidate_id', 'unknown'))}</span>"
        for record in records
    )
    return f"""
<section>
  <h2>{escape(title)}</h2>
  <p class="muted">Archive: {escape(str(archive_path))}</p>
  <div class="lineage">{lineage}</div>
  <div class="cards">{cards}</div>
</section>
"""


def _benchmark_section(results_path: Path) -> str:
    payload = _read_json(results_path)
    summary = payload.get("summary", {})
    rows = "\n".join(
        "<tr>"
        f"<td>{escape(baseline)}</td>"
        f"<td>{row.get('solve_at_budget', 0)}</td>"
        f"<td>{_ci_text(row.get('solve_at_budget_ci95'))}</td>"
        f"<td>{row.get('avg_attempts', 0)}</td>"
        f"<td>{row.get('regressions_blocked', 0)}</td>"
        f"<td>{row.get('invalid_patches_rejected', 0)}</td>"
        f"<td>{row.get('evidence_packets_admitted', 0)}</td>"
        f"<td>{row.get('unsolved_task_cost', 0)}</td>"
        f"<td>{row.get('metadata_route_matches', 0)}</td>"
        "</tr>"
        for baseline, row in summary.items()
    )
    claims = _claim_links(payload, results_path)
    return f"""
<section>
  <h2>Benchmark summary: {escape(str(payload.get('suite', 'unknown')))}</h2>
  <p class="muted">Results: {escape(str(results_path))}</p>
  {claims}
  <table>
    <thead>
      <tr>
        <th>Baseline</th>
        <th>solve@budget</th>
        <th>95% CI</th>
        <th>Avg attempts</th>
        <th>Regressions blocked</th>
        <th>Invalid rejected</th>
        <th>Evidence packets</th>
        <th>Unsolved cost</th>
        <th>Route metadata matches</th>
      </tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
</section>
"""


def _nested_search_sections(input_dir: Path, results_path: Path) -> list[str]:
    payload = _read_json(results_path)
    sections: list[str] = []
    for result in payload.get("results", []):
        archive_path = result.get("archive_path")
        if not archive_path:
            continue
        archive = Path(archive_path)
        if not archive.is_absolute():
            archive = Path.cwd() / archive
        if archive.exists():
            title = f"{result.get('case_id', 'case')} / {result.get('baseline', 'baseline')}"
            sections.append(_search_section(archive, title=title))
            if len(sections) >= 3:
                break
    return sections


def _candidate_card(record: dict[str, Any], index: int) -> str:
    status = str(record.get("status", "unknown"))
    evidence = record.get("evidence")
    gates = record.get("gates", [])
    gate_items = "\n".join(
        f"<li class='{_status_class(bool(gate.get('passed')))}'>"
        f"{escape(str(gate.get('name', 'gate')))}: {'PASS' if gate.get('passed') else 'FAIL'}"
        "</li>"
        for gate in gates
    )
    evidence_html = "<p class='muted'>No failure evidence; candidate promoted.</p>"
    if evidence:
        evidence_html = (
            "<div class='evidence'>"
            f"<strong>{escape(str(evidence.get('summary', 'failure evidence')))}</strong>"
            f"<p class='muted'>Evidence ID: {escape(str(evidence.get('evidence_id', 'unknown')))}</p>"
            f"<pre>{escape(json.dumps(evidence.get('details', {}), indent=2, sort_keys=True))}</pre>"
            "</div>"
        )
    patch_html = _patch_snippet(record.get("patch_file"))
    return f"""
<article class="candidate {escape(status)}">
  <h3>{index}. {escape(str(record.get('candidate_id', 'candidate')))} — {escape(status)}</h3>
  <dl>
    <dt>Route</dt><dd>{escape(str(record.get('route', 'unknown')))}</dd>
    <dt>Selected by</dt><dd>{escape(str(record.get('selection_reason') or 'unknown'))}</dd>
    <dt>Intent</dt><dd>{escape(str(record.get('candidate_intent') or 'unknown'))}</dd>
    <dt>Parent</dt><dd>{escape(str(record.get('parent_id') or 'none'))}</dd>
    <dt>Gen</dt><dd>{escape(str(record.get('generation', 0)))}</dd>
    <dt>Failure</dt><dd>{escape(str(record.get('failure_type') or 'none'))}</dd>
    <dt>Fingerprint</dt><dd>{escape(str(record.get('failure_fingerprint') or 'none'))}</dd>
    <dt>Score</dt><dd>{escape(str(record.get('score', 0)))}</dd>
  </dl>
  <ul class="gates">{gate_items}</ul>
  {evidence_html}
  {patch_html}
</article>
"""


def _claim_links(payload: dict[str, Any], results_path: Path) -> str:
    full = payload.get("summary", {}).get("full_gategrpo", {})
    single = payload.get("summary", {}).get("single_shot", {})
    evidence_aware = payload.get("summary", {}).get("evidence_aware_review", {})
    return f"""
  <div class="claims">
    <h3>Claim-to-artifact links</h3>
    <ul>
      <li><strong>Claim:</strong> budgeted repair search improves solve@budget over single shot.
        <br><span class="muted">Evidence: full_gategrpo={escape(str(full.get('solve_at_budget', 'n/a')))} vs single_shot={escape(str(single.get('solve_at_budget', 'n/a')))} in {escape(str(results_path))}.</span></li>
      <li><strong>Claim:</strong> bounded evidence plus route metadata is isolated as a stronger review baseline.
        <br><span class="muted">Evidence: evidence_aware_review={escape(str(evidence_aware.get('solve_at_budget', 'n/a')))} vs full_gategrpo={escape(str(full.get('solve_at_budget', 'n/a')))}; full GateGRPO adds archive, lineage, fingerprints, policy, and report artifacts.</span></li>
      <li><strong>Claim:</strong> route-aware selection uses declared candidate metadata, not patch filenames.
        <br><span class="muted">Evidence: route metadata matches={escape(str(full.get('metadata_route_matches', 0)))} plus candidate archives and controller traces.</span></li>
      <li><strong>Claim:</strong> hard gates block regressions and invalid patches before promotion.
        <br><span class="muted">Evidence: regression and invalid rejection counts in this summary plus nested candidate traces.</span></li>
    </ul>
  </div>
"""


def _ci_text(value: object) -> str:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return f"{value[0]}-{value[1]}"
    return "n/a"


def _patch_snippet(patch_file: object) -> str:
    if not patch_file:
        return ""
    path = Path(str(patch_file))
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists() or not path.is_file():
        return f"<p class='muted'>Patch file unavailable: {escape(str(patch_file))}</p>"
    content = path.read_text(encoding="utf-8")[:5000]
    return f"<details><summary>Patch diff: {escape(str(patch_file))}</summary><pre>{escape(content)}</pre></details>"


def _page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{ color-scheme: light; --ok: #d7f7df; --bad: #ffe1de; --ink: #17202a; --muted: #687385; }}
    body {{ font-family: Inter, ui-sans-serif, system-ui, sans-serif; margin: 32px; color: var(--ink); }}
    h1 {{ margin-bottom: 4px; }}
    section {{ margin: 28px 0; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #d9dee7; padding: 8px 10px; text-align: left; }}
    th {{ background: #f5f7fb; }}
    pre {{ background: #111827; color: #e5e7eb; overflow-x: auto; padding: 12px; border-radius: 8px; }}
    .muted {{ color: var(--muted); }}
    .excerpt {{ white-space: pre-wrap; }}
    .lineage {{ margin: 14px 0; line-height: 2.1; }}
    .node {{ border: 1px solid #c7ced9; border-radius: 999px; padding: 5px 9px; margin-right: 6px; }}
    .node.promoted, .candidate.promoted {{ background: var(--ok); }}
    .node.rejected, .candidate.rejected, .candidate.stopped {{ background: var(--bad); }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }}
    .candidate {{ border: 1px solid #c7ced9; border-radius: 12px; padding: 14px; }}
    .candidate dl {{ display: grid; grid-template-columns: 90px 1fr; gap: 4px 10px; }}
    .candidate dt {{ font-weight: 700; }}
    .gates .passed {{ color: #166534; }}
    .gates .failed {{ color: #991b1b; font-weight: 700; }}
    .evidence {{ border-left: 4px solid #f59e0b; padding-left: 10px; }}
    .artifact-list {{ line-height: 1.8; }}
    td.best {{ background: var(--ok); font-weight: 700; }}
  </style>
</head>
<body>
  <h1>{escape(title)}</h1>
  <p class="muted">Static report generated from GateGRPO JSON artifacts.</p>
  {body}
</body>
</html>
"""


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _status_class(passed: bool) -> str:
    return "passed" if passed else "failed"
