from ragshop.sales_consultance.application.sales_consultant_service import (
    SalesConsultantService,
)
from ragshop.sales_consultance.domain.model.product_match import ProductMatch

from tests.sales_consultance.fakes.fake_llm import FakeLLM
from tests.sales_consultance.fakes.fake_product_knowledge import FakeProductKnowledge


def test_ask_retrieves_and_passes_context_to_llm():
    matches = [
        ProductMatch(
            product_id="P001",
            name="EcoClean X200",
            description="HEPA-Staubsauger",
            category="Vacuum Cleaner",
            price="129 EUR",
        )
    ]
    llm = FakeLLM(responses=["Wir empfehlen den X200."])
    pkb = FakeProductKnowledge(fixed_matches=matches)
    service = SalesConsultantService(llm=llm, product_knowledge=pkb)

    answer = service.ask("Welcher Staubsauger ist gut?")

    assert answer == "Wir empfehlen den X200."
    assert pkb.queries == [("Welcher Staubsauger ist gut?", 3)]
    user_message = llm.calls[0][1]["content"]
    assert "EcoClean X200" in user_message
    assert "HEPA-Staubsauger" in user_message


def test_ask_renders_fallback_when_no_matches():
    llm = FakeLLM(responses=["Da habe ich leider keine Informationen."])
    pkb = FakeProductKnowledge(fixed_matches=[])
    service = SalesConsultantService(llm=llm, product_knowledge=pkb)

    service.ask("Wie ist das Wetter?")

    assert "no matching products found" in llm.calls[0][1]["content"]


def test_upsert_product_delegates_to_pkb():
    pkb = FakeProductKnowledge()
    service = SalesConsultantService(llm=FakeLLM(), product_knowledge=pkb)

    service.upsert_product({"id": "P001", "name": "X200"})

    assert pkb.upserted == {"P001": {"id": "P001", "name": "X200"}}
