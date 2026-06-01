from typing import Dict, List

from ragshop.sales_consultance.domain.model.product_match import ProductMatch
from ragshop.sales_consultance.domain.ports.product_knowledge_port import (
    ProductKnowledgePort,
)


class FakeProductKnowledge(ProductKnowledgePort):
    def __init__(self, fixed_matches: List[ProductMatch] = None):
        self._fixed = list(fixed_matches or [])
        self.upserted: Dict[str, dict] = {}
        self.queries: List[tuple] = []

    def find_matches(self, query: str, k: int) -> List[ProductMatch]:
        self.queries.append((query, k))
        return self._fixed[:k]

    def upsert_product(self, product: dict) -> None:
        self.upserted[product["id"]] = product
