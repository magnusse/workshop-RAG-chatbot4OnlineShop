import uuid
from typing import List

from .offer_request import OfferLine, OfferRequest
from .offer_store import JsonOfferStore


class OrderService:
    """Public facade of the Order Management bounded context."""

    def __init__(self, store: JsonOfferStore):
        self._store = store

    # Domain Story Step 11: Customer orders TO Shopping Cart
    def submit_offer_request(self, customer_id: str, lines: List[dict]) -> str:
        offer = OfferRequest(
            id=str(uuid.uuid4()),
            customer_id=customer_id,
            lines=[
                OfferLine(
                    product_id=line["product_id"],
                    name=line["name"],
                    quantity=line.get("quantity", 1),
                    price=line.get("price"),
                )
                for line in lines
            ],
        )
        self._store.save(offer)
        return offer.id
