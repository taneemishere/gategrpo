def parse_config(text):
    """Parse KEY=VALUE lines into a dictionary."""
    config = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=")
        config[key.strip()] = value.strip()
    return config
