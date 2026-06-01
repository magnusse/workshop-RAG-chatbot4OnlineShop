import json
from typing import Dict

from ragshop.product_information_management.bootstrap import bootstrap_pkb_from_pim
from ragshop.product_information_management.ports import ProductKnowledgeSink
from ragshop.product_information_management.product_repository import (
    JsonProductRepository,
)


class _RecordingSink(ProductKnowledgeSink):
    def __init__(self) -> None:
        self.upserted: Dict[str, dict] = {}

    def upsert_product(self, product: dict) -> None:
        self.upserted[product["id"]] = product


def test_bootstrap_pushes_every_product(tmp_path):
    products_file = tmp_path / "products.json"
    products_file.write_text(
        json.dumps(
            [
                {
                    "id": "P001",
                    "name": "X200",
                    "category": "Vacuum Cleaner",
                    "description": "A vacuum",
                    "price": "129 EUR",
                    "compatibility": [],
                },
                {
                    "id": "P002",
                    "name": "Z300",
                    "category": "Vacuum Cleaner",
                    "description": "Another vacuum",
                    "price": "159 EUR",
                    "compatibility": ["Filter"],
                },
            ]
        ),
        encoding="utf-8",
    )
    repo = JsonProductRepository(str(products_file))
    sink = _RecordingSink()

    count = bootstrap_pkb_from_pim(repo, sink)

    assert count == 2
    assert set(sink.upserted.keys()) == {"P001", "P002"}
    assert sink.upserted["P002"]["compatibility"] == ["Filter"]
