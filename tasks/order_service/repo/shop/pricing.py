DISCOUNTS = {
    "WELCOME10": 10,
    "HALF": 50,
}


def apply_discount(subtotal_cents: int, code: str | None) -> int:
    if code is None:
        return subtotal_cents
    discount_pct = DISCOUNTS[code]
    discounted = subtotal_cents * (100 - discount_pct) / 100.0
    return int(discounted)


def with_tax(amount_cents: int, tax_bps: int) -> int:
    taxed = amount_cents * (10000 + tax_bps) / 10000.0
    return int(taxed)
