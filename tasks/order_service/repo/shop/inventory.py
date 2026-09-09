class Inventory:
    def __init__(self, stock: dict[str, int]):
        self._stock = dict(stock)

    def available(self, sku: str) -> int:
        return self._stock.get(sku, 0)

    def reserve(self, sku: str, qty: int) -> None:
        self._stock[sku] -= qty
