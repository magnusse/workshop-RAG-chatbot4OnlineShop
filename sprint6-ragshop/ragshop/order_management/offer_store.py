import json
from pathlib import Path

from .offer_request import OfferRequest


class JsonOfferStore:
    """Persists offer requests as individual JSON files."""

    def __init__(self, directory: str):
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)

    # Domain Story Step 11: Customer orders TO Shopping Cart
    def save(self, offer: OfferRequest) -> None:
        payload = {
            "id": offer.id,
            "customer_id": offer.customer_id,
            "created_at": offer.created_at.isoformat(),
            "lines": [
                {
                    "product_id": line.product_id,
                    "name": line.name,
                    "quantity": line.quantity,
                    "price": line.price,
                }
                for line in offer.lines
            ],
        }
        target = self._dir / f"{offer.id}.json"
        with target.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
