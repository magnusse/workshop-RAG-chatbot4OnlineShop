from abc import ABC, abstractmethod
from typing import Optional

from ..model.conversation import Conversation, ConversationId
from ..model.customer import CustomerId


class ConversationRepository(ABC):
    @abstractmethod
    def find(self, conversation_id: ConversationId) -> Optional[Conversation]:
        ...

    @abstractmethod
    def save(self, conversation: Conversation) -> None:
        ...

    @abstractmethod
    def create(self, customer_id: CustomerId) -> Conversation:
        ...
