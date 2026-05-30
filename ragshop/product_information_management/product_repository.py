import json
from pathlib import Path
from typing import Dict, Iterator, Optional

from .product import Product


class JsonProductRepository:
    """File-backed master product repository (PIM Repo).
    Reads from / writes to a JSON file."""

    def __init__(self, file_path: str):
        self._path = Path(file_path)
        self._products: Dict[str, Product] = {}
        if self._path.exists():
            self._load()

    def _load(self) -> None:
        with self._path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        self._products = {p["id"]: Product.from_dict(p) for p in data}

    def _flush(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as f:
            json.dump(
                [p.as_dict() for p in self._products.values()],
                f,
                ensure_ascii=False,
                indent=2,
            )

    def all(self) -> Iterator[Product]:
        return iter(self._products.values())

    def get(self, product_id: str) -> Optional[Product]:
        return self._products.get(product_id)

    def save(self, product: Product) -> None:
        self._products[product.id] = product
        self._flush()

    def delete(self, product_id: str) -> None:
        if product_id in self._products:
            del self._products[product_id]
            self._flush()
