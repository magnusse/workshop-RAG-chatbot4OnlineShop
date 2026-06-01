import pytest

from ragshop.sales_consultance.domain.model.conversation import (
    Conversation,
    ConversationId,
)
from ragshop.sales_consultance.domain.model.customer import CustomerId
from ragshop.sales_consultance.domain.model.question import Answer, Question


def _conversation() -> Conversation:
    return Conversation(
        id=ConversationId(value="c1"),
        customer_id=CustomerId(value="guest"),
    )


def test_record_turn_appends_to_history():
    conv = _conversation()
    conv.record_turn(Question(text="Frage A"), Answer(text="Antwort A"))
    assert len(conv.turns) == 1


def test_history_messages_alternate_roles():
    conv = _conversation()
    conv.record_turn(Question(text="Frage A"), Answer(text="Antwort A"))
    conv.record_turn(Question(text="Frage B"), Answer(text="Antwort B"))
    history = conv.as_history_messages()
    assert [m["role"] for m in history] == ["user", "assistant", "user", "assistant"]


def test_question_rejects_empty_text():
    with pytest.raises(ValueError):
        Question(text="   ")
