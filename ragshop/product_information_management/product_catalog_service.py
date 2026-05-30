from .ports import ProductKnowledgeSink
from .product import Product
from .product_repository import JsonProductRepository


class ProductCatalogService:
    """Public facade of the PIM bounded context.
    Every catalog mutation goes through here so the Product Knowledge Base
    stays in sync with the master data."""

    def __init__(self, repository: JsonProductRepository, sink: ProductKnowledgeSink):
        self._repository = repository
        self._sink = sink

    # Domain Story Step 1: Product Manager feeds product information INTO PIM Repo
    def add_or_update(self, product: Product) -> None:
        self._repository.save(product)
        # Domain Story Step 2: PIM updates new/changed product INTO PKB
        self._sink.upsert_product(product.as_dict())

    def remove(self, product_id: str) -> None:
        self._repository.delete(product_id)
        self._sink.delete_product(product_id)

    def get(self, product_id: str):
        return self._repository.get(product_id)
