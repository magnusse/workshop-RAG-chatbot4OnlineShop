from dataclasses import dataclass, field
from typing import List, Optional

from .customer import CustomerId


@dataclass(frozen=True)
class OfferLine:
    product_id: str
    name: str
    price: Optional[str] = None
    quantity: int = 1


@dataclass(frozen=True)
class Offer:
    customer_id: CustomerId
    lines: List[OfferLine] = field(default_factory=list)

    def is_empty(self) -> bool:
        return len(self.lines) == 0
