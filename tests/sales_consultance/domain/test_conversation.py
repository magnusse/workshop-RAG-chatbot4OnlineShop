import pytest

from ragshop.sales_consultance.domain.model.conversation import (
    Conversation,
    ConversationId,
)
from ragshop.sales_consultance.domain.model.customer import CustomerId
from ragshop.sales_consultance.domain.model.intent import Intent
from ragshop.sales_consultance.domain.model.product_match import ProductMatch
from ragshop.sales_consultance.domain.model.question import Answer, Question


def _conversation() -> Conversation:
    return Conversation(
        id=ConversationId(value="c1"),
        customer_id=CustomerId(value="guest"),
    )


def test_record_turn_appends_to_history():
    conv = _conversation()
    conv.record_turn(
        Question(text="Was habt ihr an Staubsaugern?"),
        Answer(text="Wir haben den X200."),
        Intent.PRODUCT_INFO,
    )
    assert len(conv.turns) == 1
    assert conv.turns[0].intent == Intent.PRODUCT_INFO


def test_history_messages_alternate_user_and_assistant():
    conv = _conversation()
    conv.record_turn(
        Question(text="Frage A"), Answer(text="Antwort A"), Intent.PRODUCT_INFO
    )
    conv.record_turn(Question(text="Frage B"), Answer(text="Antwort B"), Intent.SMALLTALK)
    history = conv.as_history_messages()
    assert [m["role"] for m in history] == ["user", "assistant", "user", "assistant"]
    assert history[0]["content"] == "Frage A"
    assert history[3]["content"] == "Antwort B"


def test_recent_matches_deduplicates_across_turns():
    conv = _conversation()
    m1 = ProductMatch(product_id="P001", name="X", description="d")
    m2 = ProductMatch(product_id="P002", name="Y", description="d")
    conv.record_turn(
        Question(text="q1"), Answer(text="a1"), Intent.PRODUCT_INFO, matches=[m1]
    )
    conv.record_turn(
        Question(text="q2"), Answer(text="a2"), Intent.PRODUCT_INFO, matches=[m1, m2]
    )
    matches = conv.recent_matches()
    assert [m.product_id for m in matches] == ["P001", "P002"]


def test_question_rejects_empty_text():
    with pytest.raises(ValueError):
        Question(text="   ")
