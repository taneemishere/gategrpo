def parse_table(markdown):
    """Parse a simple Markdown table into row dictionaries."""
    lines = [line.strip() for line in markdown.splitlines() if line.strip()]
    table_lines = [line for line in lines if line.startswith("|")]
    if len(table_lines) < 2:
        return []

    headers = [cell.strip() for cell in table_lines[0].strip("|").split("|")]
    rows = []
    for line in table_lines[2:]:
        values = [cell.strip() for cell in line.strip("|").split("|")]
        rows.append(dict(zip(headers, values)))
    return rows
