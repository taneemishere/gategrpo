import pytest

from cart import cart_total
from pricing import discounted_cents


def test_invalid_discount_percent_raises_value_error():
    with pytest.raises(ValueError):
        discounted_cents(100, -1)
    with pytest.raises(ValueError):
        discounted_cents(100, 101)


def test_quantity_must_be_positive():
    with pytest.raises(ValueError):
        cart_total([(100, 0, 0)])


def test_discount_extremes_work():
    assert discounted_cents(123, 0) == 123
    assert discounted_cents(123, 100) == 0
