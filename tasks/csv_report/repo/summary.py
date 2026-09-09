from rows import parse_rows


def total(text, column):
    """Sum one column from a CSV-like table."""
    rows = parse_rows(text)
    if not rows:
        return 0
    column_index = 0 if column else 0
    total = 0
    for row in rows:
        if not row:
            continue
        total += int(row[column_index])
    return total
