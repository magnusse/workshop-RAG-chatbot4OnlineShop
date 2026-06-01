from ragshop.product_information_management.ports import ProductKnowledgeSink

from ..application.api import SalesConsultantApi


class PimSinkAdapter(ProductKnowledgeSink):
    """Inbound adapter: bridges PIM's ProductKnowledgeSink port to the
    Sales Consultance API. PIM stays decoupled from Sales Consultance internals."""

    def __init__(self, sales_consultant_api: SalesConsultantApi):
        self._api = sales_consultant_api

    # Domain Story Step 2: PIM updates new/changed product INTO Product Knowledge Base
    def upsert_product(self, product: dict) -> None:
        self._api.upsert_product(product)

    def delete_product(self, product_id: str) -> None:
        self._api.delete_product(product_id)
