from shop.catalog import unit_price_cents


def checkout(cart, inventory, discount_code=None, tax_bps=0):
    subtotal_cents = 0
    for sku, qty in cart:
        subtotal_cents += unit_price_cents(sku) * qty
    return {
        "subtotal_cents": subtotal_cents,
        "discount_code": discount_code,
        "total_cents": subtotal_cents,
    }
