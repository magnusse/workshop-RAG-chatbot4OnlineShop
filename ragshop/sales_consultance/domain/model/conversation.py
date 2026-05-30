from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from .customer import CustomerId
from .intent import Intent
from .product_match import ProductMatch
from .question import Answer, Question


@dataclass(frozen=True)
class ConversationId:
    value: str


@dataclass
class Turn:
    question: Question
    answer: Answer
    intent: Intent
    matches: List[ProductMatch] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Conversation:
    """Aggregate Root: a single sales consultation session with one customer."""

    id: ConversationId
    customer_id: CustomerId
    turns: List[Turn] = field(default_factory=list)

    # Domain Story Step 3: Customer clarifies questions WITH Sales Assistant
    def record_turn(
        self,
        question: Question,
        answer: Answer,
        intent: Intent,
        matches: Optional[List[ProductMatch]] = None,
    ) -> Turn:
        turn = Turn(
            question=question,
            answer=answer,
            intent=intent,
            matches=list(matches) if matches else [],
        )
        self.turns.append(turn)
        return turn

    def recent_matches(self, limit: int = 5) -> List[ProductMatch]:
        """Return the most recently mentioned product matches across turns,
        used when building an offer from conversation context."""
        seen = set()
        result: List[ProductMatch] = []
        for turn in reversed(self.turns):
            for match in turn.matches:
                if match.product_id in seen:
                    continue
                seen.add(match.product_id)
                result.append(match)
                if len(result) >= limit:
                    return result
        return result

    def as_history_messages(self) -> List[dict]:
        """Render the conversation as OpenAI-style chat messages
        (no system prompt, no current-turn context — that's the caller's job)."""
        messages: List[dict] = []
        for turn in self.turns:
            messages.append({"role": "user", "content": turn.question.text})
            messages.append({"role": "assistant", "content": turn.answer.text})
        return messages
