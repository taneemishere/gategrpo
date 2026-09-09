# CSV report spec

- `parse_rows(text)` returns one list per non-blank input line.
- Split each line on commas.
- Fields may be wrapped in double quotes; quoted fields may contain commas that are not separators.
- Surrounding whitespace outside a field is trimmed.
- Quoted fields are returned without the surrounding quotes.
- Empty fields are preserved as empty strings.
- `total(text, column)` treats the first parsed row as the header, sums the named column across body rows, and raises `ValueError` if the column is missing or any body value is not a valid integer.
