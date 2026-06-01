from typing import List

from ragshop.sales_consultance.domain.model.offer import Offer
from ragshop.sales_consultance.domain.ports.order_management_port import (
    OfferReceipt,
    OrderManagementPort,
)


class FakeOrderManagement(OrderManagementPort):
    def __init__(self) -> None:
        self.submitted: List[Offer] = []

    def submit_offer_request(self, offer: Offer) -> OfferReceipt:
        self.submitted.append(offer)
        return OfferReceipt(offer_id=f"offer-{len(self.submitted)}")
