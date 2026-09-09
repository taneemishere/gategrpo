# Config loader spec

- `parse_config(text)` returns a dictionary of key/value pairs.
- Blank lines and lines starting with `#` are ignored.
- Keys and values are stripped of surrounding whitespace.
- Values may contain `=`; only the first equals sign separates key and value.
- Single-quoted or double-quoted values have the outer quotes removed.
- Duplicate keys are malformed and must raise `ValueError`.
