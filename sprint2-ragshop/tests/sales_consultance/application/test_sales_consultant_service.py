from ragshop.sales_consultance.application.sales_consultant_service import (
    SalesConsultantService,
)

from tests.sales_consultance.fakes.fake_llm import FakeLLM


def test_ask_calls_llm_with_system_prompt_and_question():
    llm = FakeLLM(responses=["Dazu empfehle ich den X200."])
    service = SalesConsultantService(llm=llm)

    answer = service.ask("Welcher Staubsauger ist gut?")

    assert answer == "Dazu empfehle ich den X200."
    assert len(llm.calls) == 1
    messages = llm.calls[0]
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "Welcher Staubsauger ist gut?"}


def test_each_question_is_independent():
    llm = FakeLLM(responses=["Antwort 1", "Antwort 2"])
    service = SalesConsultantService(llm=llm)

    service.ask("Frage 1")
    service.ask("Frage 2")

    # No history yet — each call sends only system + the single user turn.
    assert all(len(call) == 2 for call in llm.calls)
