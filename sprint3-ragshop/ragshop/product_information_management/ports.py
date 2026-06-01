from abc import ABC, abstractmethod


class ProductKnowledgeSink(ABC):
    """Outbound interface used by PIM to push catalog changes into the
    Sales Consultance bounded context.

    Sprint 3 surface: only upsert. delete is added in Sprint 5 once we
    have a use case for taking products out of the catalogue at runtime.
    """

    # Domain Story Step 2: PIM updates new/changed product INTO Product Knowledge Base
    @abstractmethod
    def upsert_product(self, product: dict) -> None:
        ...
