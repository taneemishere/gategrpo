def parse_ranges(text):
    """Parse comma-separated integers into a list."""
    values = []
    for part in text.split(","):
        values.append(int(part))
    return values
