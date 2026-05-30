from .ports import ProductKnowledgeSink
from .product_repository import JsonProductRepository


def bootstrap_pkb_from_pim(
    repository: JsonProductRepository, sink: ProductKnowledgeSink
) -> int:
    """Push every product currently in the PIM repo into the PKB.
    Idempotent: an upsert against an existing entry just refreshes it."""
    count = 0
    for product in repository.all():
        # Domain Story Step 2: PIM updates new/changed product INTO PKB
        sink.upsert_product(product.as_dict())
        count += 1
    return count
