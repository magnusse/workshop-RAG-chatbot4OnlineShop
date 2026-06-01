from abc import ABC, abstractmethod


class ProductKnowledgeSink(ABC):
    """Outbound interface used by PIM to push catalog changes into the
    Sales Consultance bounded context. Wired at composition time."""

    # Domain Story Step 2: PIM updates new/changed product INTO Product Knowledge Base
    @abstractmethod
    def upsert_product(self, product: dict) -> None:
        ...

    @abstractmethod
    def delete_product(self, product_id: str) -> None:
        ...
