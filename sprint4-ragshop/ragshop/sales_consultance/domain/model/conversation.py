from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from .customer import CustomerId
from .product_match import ProductMatch
from .question import Answer, Question


@dataclass(frozen=True)
class ConversationId:
    value: str


@dataclass
class Turn:
    question: Question
    answer: Answer
    matches: List[ProductMatch] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Conversation:
    """Aggregate Root: a sales consultation session with one customer.
    Introduced in Sprint 4 because we now need state across turns."""

    id: ConversationId
    customer_id: CustomerId
    turns: List[Turn] = field(default_factory=list)

    # Domain Story Step 3 (repeated): Customer clarifies questions WITH Sales Assistant
    def record_turn(
        self,
        question: Question,
        answer: Answer,
        matches: Optional[List[ProductMatch]] = None,
    ) -> Turn:
        turn = Turn(
            question=question,
            answer=answer,
            matches=list(matches) if matches else [],
        )
        self.turns.append(turn)
        return turn

    def as_history_messages(self) -> List[dict]:
        """Render the conversation as OpenAI-style chat messages
        (no system prompt, no current-turn context — that's the caller's job)."""
        messages: List[dict] = []
        for turn in self.turns:
            messages.append({"role": "user", "content": turn.question.text})
            messages.append({"role": "assistant", "content": turn.answer.text})
        return messages
