from abc import ABC, abstractmethod
from typing import List

from ..model.product_match import ProductMatch


class ProductKnowledgePort(ABC):
    """Outbound port to the Product Knowledge Base.

    Read side (find_matches) is used during sales consultation.
    Write side (upsert/delete) is called by the inbound API when PIM pushes
    catalog changes — see Domain Story Step 2.
    """

    @abstractmethod
    def find_matches(self, query: str, k: int) -> List[ProductMatch]:
        ...

    @abstractmethod
    def upsert_product(self, product: dict) -> None:
        """Insert or replace a product entry in the knowledge base.
        Accepts a plain dict (a transport DTO) to keep PIM and Sales Consultance
        decoupled — the adapter decides how to chunk / embed."""

    @abstractmethod
    def delete_product(self, product_id: str) -> None:
        ...
