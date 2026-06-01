import json
from pathlib import Path
from typing import Dict, Iterator, Optional

from .product import Product


class JsonProductRepository:
    """File-backed master product repository (PIM Repo). Reads from a JSON file.
    Sprint 3 is read-only — there's no runtime mutation yet. Save/delete come in Sprint 5."""

    def __init__(self, file_path: str):
        self._path = Path(file_path)
        self._products: Dict[str, Product] = {}
        if self._path.exists():
            self._load()

    def _load(self) -> None:
        with self._path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        self._products = {p["id"]: Product.from_dict(p) for p in data}

    def all(self) -> Iterator[Product]:
        return iter(self._products.values())

    def get(self, product_id: str) -> Optional[Product]:
        return self._products.get(product_id)
