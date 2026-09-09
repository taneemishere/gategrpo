# Slugifier spec

- `slugify(text)` returns a lowercase URL slug.
- Letters and digits are preserved.
- Whitespace and hyphen runs collapse into a single hyphen.
- Punctuation and other symbols are removed.
- Leading and trailing hyphens are stripped.
- Inputs with no slug characters return an empty string.
