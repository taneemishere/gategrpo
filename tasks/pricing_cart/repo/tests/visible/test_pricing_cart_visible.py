from cart import cart_total
from pricing import discounted_cents


def test_discounted_cents_rounds_half_up_and_cart_total_uses_discount():
    assert discounted_cents(105, 10) == 95
    assert cart_total([(105, 2, 10)]) == 190


def test_simple_discount_and_cart_total_case():
    assert discounted_cents(200, 0) == 200
    assert cart_total([(200, 3, 0)]) == 600
