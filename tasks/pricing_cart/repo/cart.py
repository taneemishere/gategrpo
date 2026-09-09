from pricing import discounted_cents


def cart_total(items):
    """Return the discounted cart total in cents."""
    total = 0
    for unit_cents, quantity, discount_pct in items:
        total += unit_cents * quantity
    return total
