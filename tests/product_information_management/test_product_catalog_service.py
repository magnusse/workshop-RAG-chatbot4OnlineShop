from typing import Dict, List

from ragshop.product_information_management.bootstrap import bootstrap_pkb_from_pim
from ragshop.product_information_management.ports import ProductKnowledgeSink
from ragshop.product_information_management.product import Product
from ragshop.product_information_management.product_catalog_service import (
    ProductCatalogService,
)
from ragshop.product_information_management.product_repository import (
    JsonProductRepository,
)


class _RecordingSink(ProductKnowledgeSink):
    def __init__(self) -> None:
        self.upserted: Dict[str, dict] = {}
        self.deleted: List[str] = []

    def upsert_product(self, product: dict) -> None:
        self.upserted[product["id"]] = product

    def delete_product(self, product_id: str) -> None:
        self.deleted.append(product_id)


def _make_product(pid: str = "P999") -> Product:
    return Product(
        id=pid,
        name="Test Geraet",
        category="Mixer",
        description="Beschreibung",
        price="99 EUR",
        source="Manufacturer",
        upddate="2026-05-01",
        delflag=False,
        prodcatversion="1.0",
        compatibility=["Zubehoer A"],
    )


def test_add_or_update_persists_and_pushes_to_sink(tmp_path):
    repo = JsonProductRepository(str(tmp_path / "products.json"))
    sink = _RecordingSink()
    service = ProductCatalogService(repository=repo, sink=sink)

    service.add_or_update(_make_product("P001"))

    assert repo.get("P001") is not None
    assert "P001" in sink.upserted
    assert sink.upserted["P001"]["name"] == "Test Geraet"


def test_remove_deletes_in_repo_and_sink(tmp_path):
    repo = JsonProductRepository(str(tmp_path / "products.json"))
    sink = _RecordingSink()
    service = ProductCatalogService(repository=repo, sink=sink)
    service.add_or_update(_make_product("P002"))

    service.remove("P002")

    assert repo.get("P002") is None
    assert sink.deleted == ["P002"]


def test_bootstrap_pushes_all_products(tmp_path):
    file_path = tmp_path / "products.json"
    repo = JsonProductRepository(str(file_path))
    sink = _RecordingSink()
    service = ProductCatalogService(repository=repo, sink=sink)
    service.add_or_update(_make_product("P001"))
    service.add_or_update(_make_product("P002"))
    sink.upserted.clear()

    count = bootstrap_pkb_from_pim(repo, sink)

    assert count == 2
    assert set(sink.upserted.keys()) == {"P001", "P002"}
