from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ProductMatch:
    """A single product the retriever found relevant for a query.
    Introduced in Sprint 3 because we now return structured data from the
    PKB (not just raw strings) — the application needs name/price for the prompt."""

    product_id: str
    name: str
    description: str
    category: Optional[str] = None
    price: Optional[str] = None
