from abc import ABC, abstractmethod
from typing import List


class LLMPort(ABC):
    """Outbound port to a chat LLM."""

    @abstractmethod
    def generate(self, messages: List[dict]) -> str:
        ...
