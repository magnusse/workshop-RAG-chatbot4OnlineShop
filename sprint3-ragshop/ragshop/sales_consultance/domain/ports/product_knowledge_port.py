from abc import ABC, abstractmethod
from typing import List

from ..model.product_match import ProductMatch


class ProductKnowledgePort(ABC):
    """Outbound port to the Product Knowledge Base.

    Sprint 3 surface: read (find_matches) and bulk-write (upsert_product) —
    just enough to bootstrap from PIM and to retrieve at query time.
    Sprint 5 will extend with delete_product and metadata-based filtering.
    """

    # Domain Story Step 5: select best content from Product Knowledge Base
    @abstractmethod
    def find_matches(self, query: str, k: int) -> List[ProductMatch]:
        ...

    # Domain Story Step 2: PIM updates new/changed product INTO Product Knowledge Base
    @abstractmethod
    def upsert_product(self, product: dict) -> None:
        """Insert or replace a product entry. Accepts a plain dict (a DTO)
        so PIM stays decoupled from the adapter's internal chunking strategy."""
