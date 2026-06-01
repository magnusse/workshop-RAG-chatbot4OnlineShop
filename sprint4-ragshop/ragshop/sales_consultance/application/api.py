from abc import ABC, abstractmethod

from ..domain.model.conversation import ConversationId
from ..domain.model.customer import CustomerId


class SalesConsultantApi(ABC):
    """Inbound (driving) port. Sprint 4 adds a conversation lifecycle:
    callers first start a conversation, then ask within it."""

    @abstractmethod
    def start_conversation(self, customer_id: CustomerId) -> ConversationId:
        ...

    # Domain Story Step 3: Customer clarifies questions WITH Sales Assistant
    @abstractmethod
    def ask(self, conversation_id: ConversationId, question_text: str) -> str:
        ...

    # Domain Story Step 2: PIM updates new/changed product INTO Product Knowledge Base
    @abstractmethod
    def upsert_product(self, product: dict) -> None:
        ...
