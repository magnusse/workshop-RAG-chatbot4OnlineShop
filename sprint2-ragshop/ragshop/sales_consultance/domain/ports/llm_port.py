from abc import ABC, abstractmethod
from typing import List


class LLMPort(ABC):
    """Outbound port to a chat LLM. Introduced now (Sprint 2) because we want
    to test the SalesConsultantService without hitting the real WPS endpoint."""

    @abstractmethod
    def generate(self, messages: List[dict]) -> str:
        """Run a chat completion. `messages` is a list of {"role","content"} dicts
        in OpenAI format. Returns the assistant's reply text."""
