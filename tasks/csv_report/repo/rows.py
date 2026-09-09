def parse_rows(text):
    """Parse a CSV-like text table into rows."""
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        rows.append([cell.strip() for cell in line.split(',')])
    return rows
