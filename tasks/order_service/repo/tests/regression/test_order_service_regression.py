import pytest

from shop.inventory import Inventory
from shop.pricing import apply_discount, with_tax


def test_unknown_discount_code_raises_value_error():
    with pytest.raises(ValueError):
        apply_discount(1000, "NOT_A_CODE")


def test_negative_tax_bps_raises_value_error():
    with pytest.raises(ValueError):
        with_tax(1000, -1)


def test_reserving_more_than_available_raises_value_error():
    inventory = Inventory({"APPLE": 1})

    with pytest.raises(ValueError):
        inventory.reserve("APPLE", 2)


def test_half_up_rounding_cases_are_correct():
    assert apply_discount(105, "WELCOME10") == 95
    assert with_tax(250, 825) == 271
