CATALOG = {
    "APPLE": 150,
    "BREAD": 200,
    "MILK": 250,
}


def unit_price_cents(sku: str) -> int:
    try:
        return CATALOG[sku]
    except KeyError as exc:
        raise ValueError(f"unknown sku: {sku}") from exc
