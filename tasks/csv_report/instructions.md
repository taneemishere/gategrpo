# CSV report repair

Repair `rows.py` and `summary.py` so they follow `spec.md`.

Constraints:

- Preserve the public functions `parse_rows(text: str) -> list[list[str]]` and `total(text: str, column: str) -> int`.
- Do not modify tests or spec files.
- Keep the patch scoped to `rows.py` and `summary.py`.
