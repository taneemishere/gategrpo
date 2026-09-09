# Integer range parser spec

- `parse_ranges(text)` returns integers described by a comma-separated string.
- A single integer token adds that integer.
- A token like `2-4` expands inclusively to `2, 3, 4`.
- Whitespace around tokens is ignored.
- Duplicate integers are removed while preserving first-seen order.
- A descending range such as `4-2` is malformed and must raise `ValueError`.
