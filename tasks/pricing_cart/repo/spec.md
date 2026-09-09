# Pricing cart spec

- `discounted_cents(unit_cents, discount_pct)` applies `discount_pct` percent off `unit_cents` and rounds HALF UP to the nearest cent.
- `discount_pct` must be an int in the inclusive range 0..100; otherwise raise `ValueError`.
- `cart_total(items)` sums `discounted_cents(unit_cents, discount_pct) * quantity` for each item tuple.
- `quantity` must be a positive int; otherwise raise `ValueError`.
