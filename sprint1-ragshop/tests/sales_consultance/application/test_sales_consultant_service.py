from ragshop.sales_consultance.application.sales_consultant_service import (
    SalesConsultantService,
)


def test_returns_fallback_answer_for_any_question():
    service = SalesConsultantService()
    answer = service.ask("Wie viel kostet der Staubsauger X?")
    assert "+49" in answer


def test_same_answer_for_different_questions():
    service = SalesConsultantService()
    assert service.ask("Frage A") == service.ask("Frage B")
