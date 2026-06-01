from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass(frozen=True)
class OfferLine:
    product_id: str
    name: str
    quantity: int = 1
    price: Optional[str] = None


@dataclass(frozen=True)
class OfferRequest:
    """The offer request that arrives from Sales Consultance.
    Corresponds to the Shopping Cart entry in the domain story."""

    id: str
    customer_id: str
    lines: List[OfferLine]
    created_at: datetime = field(default_factory=datetime.utcnow)
