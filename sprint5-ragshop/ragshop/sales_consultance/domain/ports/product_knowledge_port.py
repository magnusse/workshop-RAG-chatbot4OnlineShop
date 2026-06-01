from abc import ABC, abstractmethod
from typing import List

from ..model.product_match import ProductMatch


class ProductKnowledgePort(ABC):
    """Outbound port to the Product Knowledge Base.
    Sprint 5 adds delete and the metadata-based retrieval filter."""

    # Domain Story Step 5: select best content from Product Knowledge Base
    @abstractmethod
    def find_matches(self, query: str, k: int) -> List[ProductMatch]:
        ...

    # Domain Story Step 2: PIM updates new/changed product INTO Product Knowledge Base
    @abstractmethod
    def upsert_product(self, product: dict) -> None:
        ...

    @abstractmethod
    def delete_product(self, product_id: str) -> None:
        ...
