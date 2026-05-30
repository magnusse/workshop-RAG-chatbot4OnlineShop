import json

from ragshop.order_management.offer_store import JsonOfferStore
from ragshop.order_management.order_service import OrderService


def test_submit_offer_request_persists_json(tmp_path):
    store = JsonOfferStore(directory=str(tmp_path))
    service = OrderService(store=store)

    offer_id = service.submit_offer_request(
        customer_id="guest",
        lines=[
            {"product_id": "P001", "name": "X200", "price": "129 EUR", "quantity": 1},
            {"product_id": "P002", "name": "Z300", "price": "159 EUR", "quantity": 2},
        ],
    )

    files = list(tmp_path.iterdir())
    assert len(files) == 1
    assert files[0].name == f"{offer_id}.json"

    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["customer_id"] == "guest"
    assert [line["product_id"] for line in payload["lines"]] == ["P001", "P002"]
    assert payload["lines"][1]["quantity"] == 2
