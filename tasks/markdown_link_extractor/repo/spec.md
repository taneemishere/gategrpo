# Markdown inline link extraction spec

- `extract_links(text)` returns one dictionary per inline Markdown link in order of appearance.
- Each dictionary has the shape `{"text": ..., "url": ...}`.
- Link text is the substring inside the outer `[` and `]`.
- URL is the substring inside the outer `(` and `)`.
- Link text may contain balanced square brackets.
- URL may contain balanced parentheses.
- Escaped delimiters `\[`, `\]`, `\(`, `\)` are literal characters, not structural delimiters.
- Plain text that is not part of a link is ignored.
- A link-opening `[` with no matching structural `]` / `(...)` is malformed and must raise `ValueError`.
