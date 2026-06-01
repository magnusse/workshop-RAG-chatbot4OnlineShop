from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..model.offer import Offer


@dataclass(frozen=True)
class OfferReceipt:
    offer_id: str


class OrderManagementPort(ABC):
    """Outbound port to the Order Management bounded context."""

    # Domain Story Step 10: Sales Assistant creates & sends Offer Request
    @abstractmethod
    def submit_offer_request(self, offer: Offer) -> OfferReceipt:
        ...
