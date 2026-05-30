from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ProductMatch:
    product_id: str
    name: str
    description: str
    category: Optional[str] = None
    price: Optional[str] = None
