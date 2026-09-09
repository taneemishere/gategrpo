# Markdown table parser spec

- `parse_table(markdown)` returns one dictionary per body row.
- Header names come from the first table row.
- The second table row is the separator and is not returned.
- Leading and trailing whitespace around cell values is ignored.
- Empty cells are preserved as empty strings.
- Escaped pipes (`\|`) are literal pipe characters inside a cell, not cell separators.
- A body row with a different cell count than the header is malformed and must raise `ValueError`.
