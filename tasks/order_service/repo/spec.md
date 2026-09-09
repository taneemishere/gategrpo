# Order service spec

The package `shop` provides a small checkout flow.

## Catalog

- `shop.catalog.CATALOG` contains the fixed product prices:
  - `APPLE`: 150 cents
  - `BREAD`: 200 cents
  - `MILK`: 250 cents
- `shop.catalog.unit_price_cents(sku)` returns the catalog price for `sku`.
- Unknown SKUs raise `ValueError`.

## Pricing

- `shop.pricing.DISCOUNTS` maps discount codes to percentage discounts.
- `apply_discount(subtotal_cents, code)` returns the subtotal unchanged when `code` is `None`.
- A known discount code applies that percent off and rounds HALF UP to the nearest cent.
- Unknown discount codes raise `ValueError`.
- `with_tax(amount_cents, tax_bps)` adds tax in basis points and rounds HALF UP to the nearest cent.
- `tax_bps` must be a non-negative integer; otherwise raise `ValueError`.

## Inventory

- `Inventory(stock)` stores the available quantity of each SKU.
- `available(sku)` returns the current stock level or zero for an unknown SKU.
- `reserve(sku, qty)` decrements stock when `sku` exists, `qty` is a positive integer, and enough stock is available.
- Unknown SKUs, non-positive quantities, and insufficient stock raise `ValueError`.

## Checkout

- `checkout(cart, inventory, discount_code=None, tax_bps=0)` accepts a list of `(sku, qty)` pairs.
- It computes the subtotal from the catalog.
- It reserves each cart line in inventory.
- It applies the discount, then tax, to produce the final total.
- It returns `{"subtotal_cents": ..., "discount_code": ..., "total_cents": ...}`.
