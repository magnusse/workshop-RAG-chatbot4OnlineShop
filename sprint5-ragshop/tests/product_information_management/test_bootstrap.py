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

    def delete_product(self, product_id: str) -> None:
        pass


def _full(pid: str, name: str, compat=None) -> dict:
    return {
        "id": pid,
        "name": name,
        "category": "Vacuum Cleaner",
        "description": f"{name} description",
        "price": "129 EUR",
        "source": "Manufacturer",
        "upddate": "2026-05-01",
        "delflag": False,
        "prodcatversion": "1.0",
        "compatibility": compat or [],
    }


def test_bootstrap_pushes_every_product(tmp_path):
    products_file = tmp_path / "products.json"
    products_file.write_text(
        json.dumps(
            [
                _full("P001", "X200"),
                _full("P002", "Z300", ["Filter"]),
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
