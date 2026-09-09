def discounted_cents(unit_cents, discount_pct):
    """Return the discounted unit price in cents."""
    return int(unit_cents * (100 - discount_pct) / 100)
