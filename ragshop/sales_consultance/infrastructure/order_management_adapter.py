from ..domain.model.offer import Offer
from ..domain.ports.order_management_port import OfferReceipt, OrderManagementPort
from ragshop.order_management.order_service import OrderService


class OrderManagementAdapter(OrderManagementPort):
    """Outbound adapter that calls the Order Management bounded context directly."""

    def __init__(self, order_service: OrderService):
        self._order_service = order_service

    # Domain Story Step 10: Sales Assistant creates & sends Offer Request
    def submit_offer_request(self, offer: Offer) -> OfferReceipt:
        offer_id = self._order_service.submit_offer_request(
            customer_id=offer.customer_id.value,
            lines=[
                {
                    "product_id": line.product_id,
                    "name": line.name,
                    "price": line.price,
                    "quantity": line.quantity,
                }
                for line in offer.lines
            ],
        )
        return OfferReceipt(offer_id=offer_id)
