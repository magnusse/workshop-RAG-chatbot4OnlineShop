from ragshop.sales_consultance.application.sales_consultant_service import (
    SalesConsultantService,
)
from ragshop.sales_consultance.domain.model.customer import CustomerId
from ragshop.sales_consultance.domain.model.product_match import ProductMatch
from ragshop.sales_consultance.infrastructure.in_memory_conversation_repository import (
    InMemoryConversationRepository,
)

from tests.sales_consultance.fakes.fake_llm import FakeLLM
from tests.sales_consultance.fakes.fake_product_knowledge import FakeProductKnowledge


def _build(llm, pkb):
    repo = InMemoryConversationRepository()
    service = SalesConsultantService(llm=llm, product_knowledge=pkb, conversations=repo)
    conv_id = service.start_conversation(CustomerId(value="guest"))
    return service, conv_id, repo


def test_first_question_skips_rewrite_and_retrieves_with_raw_query():
    matches = [ProductMatch(product_id="P001", name="X200", description="HEPA", price="129 EUR")]
    # On the first turn the history is empty -> rewriter returns the query as-is.
    # So the LLM is called only for the final answer.
    llm = FakeLLM(responses=["Wir empfehlen den X200."])
    pkb = FakeProductKnowledge(fixed_matches=matches)
    service, conv_id, repo = _build(llm, pkb)

    reply = service.ask(conv_id, "Welcher Staubsauger ist gut?")

    assert reply == "Wir empfehlen den X200."
    assert pkb.queries == [("Welcher Staubsauger ist gut?", 3)]
    conv = repo.find(conv_id)
    assert len(conv.turns) == 1


def test_follow_up_uses_rewritten_query_for_retrieval():
    matches = [ProductMatch(product_id="P001", name="X200", description="HEPA", price="129 EUR")]
    # Turn 1: history empty -> rewrite skipped -> 1 LLM call.
    # Turn 2: history non-empty -> rewrite called -> then answer -> 2 LLM calls.
    llm = FakeLLM(responses=[
        "Erzaehle vom X200",                    # turn 1 answer
        "Gibt es den X200 auch in weiss?",      # turn 2 rewrite
        "Ja, in weiss verfuegbar.",             # turn 2 answer
    ])
    pkb = FakeProductKnowledge(fixed_matches=matches)
    service, conv_id, _ = _build(llm, pkb)

    service.ask(conv_id, "Erzaehle mir vom X200")
    service.ask(conv_id, "Und in weiss?")

    assert [q for q, _ in pkb.queries] == [
        "Erzaehle mir vom X200",
        "Gibt es den X200 auch in weiss?",
    ]


def test_ask_records_turn_in_conversation():
    pkb = FakeProductKnowledge(fixed_matches=[])
    llm = FakeLLM(responses=["Antwort 1"])
    service, conv_id, repo = _build(llm, pkb)

    service.ask(conv_id, "Frage 1")

    conv = repo.find(conv_id)
    assert conv.turns[0].question.text == "Frage 1"
    assert conv.turns[0].answer.text == "Antwort 1"


def test_ask_raises_for_unknown_conversation():
    from ragshop.sales_consultance.domain.model.conversation import ConversationId

    service, _, _ = _build(FakeLLM(), FakeProductKnowledge())
    with __import__("pytest").raises(LookupError):
        service.ask(ConversationId(value="ghost"), "hi")


def test_upsert_product_still_delegates():
    pkb = FakeProductKnowledge()
    service, _, _ = _build(FakeLLM(), pkb)
    service.upsert_product({"id": "P001", "name": "X200"})
    assert pkb.upserted == {"P001": {"id": "P001", "name": "X200"}}
