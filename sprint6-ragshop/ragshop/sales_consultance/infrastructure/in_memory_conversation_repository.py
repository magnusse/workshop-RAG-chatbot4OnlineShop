import uuid
from typing import Dict, Optional

from ..domain.model.conversation import Conversation, ConversationId
from ..domain.model.customer import CustomerId
from ..domain.ports.conversation_repository import ConversationRepository


class InMemoryConversationRepository(ConversationRepository):
    def __init__(self) -> None:
        self._store: Dict[str, Conversation] = {}

    def find(self, conversation_id: ConversationId) -> Optional[Conversation]:
        return self._store.get(conversation_id.value)

    def save(self, conversation: Conversation) -> None:
        self._store[conversation.id.value] = conversation

    def create(self, customer_id: CustomerId) -> Conversation:
        conversation = Conversation(
            id=ConversationId(value=str(uuid.uuid4())),
            customer_id=customer_id,
        )
        self.save(conversation)
        return conversation
