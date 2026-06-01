import pytest

from ragshop.sales_consultance.application.sales_consultant_service import (
    SalesConsultantService,
)
from ragshop.sales_consultance.domain.model.customer import CustomerId
from ragshop.sales_consultance.domain.model.intent import Intent
from ragshop.sales_consultance.domain.model.product_match import ProductMatch
from ragshop.sales_consultance.infrastructure.in_memory_conversation_repository import (
    InMemoryConversationRepository,
)

from tests.sales_consultance.fakes.fake_llm import FakeLLM
from tests.sales_consultance.fakes.fake_order_management import FakeOrderManagement
from tests.sales_consultance.fakes.fake_product_knowledge import FakeProductKnowledge


def _build(llm: FakeLLM, pkb: FakeProductKnowledge, orders: FakeOrderManagement):
    repo = InMemoryConversationRepository()
    service = SalesConsultantService(
        llm=llm,
        product_knowledge=pkb,
        order_management=orders,
        conversations=repo,
    )
    conv_id = service.start_conversation(CustomerId(value="guest"))
    return service, conv_id, repo


def test_product_info_intent_runs_retrieval_and_returns_llm_answer():
    matches = [
        ProductMatch(
            product_id="P001",
            name="EcoClean X200",
            description="HEPA-Staubsauger",
            category="Vacuum Cleaner",
            price="129 EUR",
        )
    ]
    # On the first turn the query rewrite is skipped (empty history),
    # so the LLM gets only two calls: intent detection + final answer.
    llm = FakeLLM(responses=[Intent.PRODUCT_INFO.value, "Wir empfehlen den X200."])
    pkb = FakeProductKnowledge(fixed_matches=matches)

    service, conv_id, repo = _build(llm, pkb, FakeOrderManagement())

    reply = service.ask(conv_id, "Was habt ihr an Staubsaugern?")

    assert reply == "Wir empfehlen den X200."
    assert pkb.queries == [("Was habt ihr an Staubsaugern?", 3)]
    conv = repo.find(conv_id)
    assert len(conv.turns) == 1
    assert conv.turns[0].intent == Intent.PRODUCT_INFO
    assert conv.turns[0].matches == matches


def test_smalltalk_intent_skips_retrieval():
    llm = FakeLLM(responses=[Intent.SMALLTALK.value, "Hallo, gerne helfe ich Ihnen!"])
    pkb = FakeProductKnowledge()

    service, conv_id, _ = _build(llm, pkb, FakeOrderManagement())

    reply = service.ask(conv_id, "Hallo!")

    assert reply == "Hallo, gerne helfe ich Ihnen!"
    assert pkb.queries == []


def test_offer_request_submits_to_order_management():
    candidate = ProductMatch(
        product_id="P001",
        name="EcoClean X200",
        description="HEPA",
        price="129 EUR",
    )

    # 1st turn: PRODUCT_INFO so a match enters the conversation context.
    # 2nd turn: OFFER_REQUEST -> intent + extraction of P001 -> submit.
    llm = FakeLLM(responses=[
        Intent.PRODUCT_INFO.value,
        "Hier ist der X200.",
        Intent.OFFER_REQUEST.value,
        "P001",  # extraction returns this product
    ])
    pkb = FakeProductKnowledge(fixed_matches=[candidate])
    orders = FakeOrderManagement()

    service, conv_id, _ = _build(llm, pkb, orders)

    service.ask(conv_id, "Erzaehle mir vom X200")
    reply = service.ask(conv_id, "Bitte erstelle mir ein Angebot dazu")

    assert len(orders.submitted) == 1
    offer = orders.submitted[0]
    assert [line.product_id for line in offer.lines] == ["P001"]
    assert "offer-1" in reply
    assert "EcoClean X200" in reply


def test_offer_request_without_candidates_asks_back():
    llm = FakeLLM(responses=[Intent.OFFER_REQUEST.value])
    pkb = FakeProductKnowledge()
    orders = FakeOrderManagement()

    service, conv_id, _ = _build(llm, pkb, orders)
    reply = service.ask(conv_id, "Bitte ein Angebot")

    assert orders.submitted == []
    assert "Welches Produkt" in reply


def test_query_is_rewritten_on_follow_up_turn():
    matches = [
        ProductMatch(product_id="P001", name="X200", description="d", price="129 EUR")
    ]
    llm = FakeLLM(responses=[
        Intent.PRODUCT_INFO.value, "Antwort 1",            # turn 1
        Intent.PRODUCT_INFO.value,                          # intent on turn 2
        "Gibt es den X200 auch in weiss?",                  # rewrite on turn 2
        "Ja, in weiss verfuegbar.",                         # answer on turn 2
    ])
    pkb = FakeProductKnowledge(fixed_matches=matches)

    service, conv_id, _ = _build(llm, pkb, FakeOrderManagement())

    service.ask(conv_id, "Erzaehle mir vom X200")
    service.ask(conv_id, "Und in weiss?")

    # Two retrievals, second one used the rewritten query
    assert [q for q, _ in pkb.queries] == [
        "Erzaehle mir vom X200",
        "Gibt es den X200 auch in weiss?",
    ]


def test_upsert_product_delegates_to_pkb():
    pkb = FakeProductKnowledge()
    service, _, _ = _build(FakeLLM(responses=[]), pkb, FakeOrderManagement())

    service.upsert_product({"id": "P001", "name": "X200"})

    assert pkb.upserted == {"P001": {"id": "P001", "name": "X200"}}


def test_ask_raises_for_unknown_conversation():
    service, _, _ = _build(FakeLLM(responses=[]), FakeProductKnowledge(), FakeOrderManagement())
    from ragshop.sales_consultance.domain.model.conversation import ConversationId

    with pytest.raises(LookupError):
        service.ask(ConversationId(value="does-not-exist"), "hi")
