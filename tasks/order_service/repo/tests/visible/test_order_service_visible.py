from shop.checkout import checkout
from shop.inventory import Inventory


def test_checkout_applies_discount_tax_and_reserves_inventory():
    inventory = Inventory({"APPLE": 5, "BREAD": 5, "MILK": 5})
    cart = [("APPLE", 2), ("BREAD", 1)]

    result = checkout(cart, inventory, discount_code="HALF", tax_bps=825)

    assert result["subtotal_cents"] == 500
    assert result["discount_code"] == "HALF"
    assert result["total_cents"] == 271
    assert inventory.available("APPLE") == 3
    assert inventory.available("BREAD") == 4


def test_checkout_handles_simple_order_without_discount_or_tax():
    inventory = Inventory({"APPLE": 3, "BREAD": 2, "MILK": 1})
    cart = [("MILK", 1)]

    result = checkout(cart, inventory)

    assert result == {"subtotal_cents": 250, "discount_code": None, "total_cents": 250}
    assert inventory.available("MILK") == 0
