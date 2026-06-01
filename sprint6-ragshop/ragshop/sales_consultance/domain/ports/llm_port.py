from abc import ABC, abstractmethod
from typing import List


class LLMPort(ABC):
    """Outbound port to a chat LLM. Adapters implement this against
    a concrete provider (e.g. WPS-hosted Mistral)."""

    @abstractmethod
    def generate(self, messages: List[dict]) -> str:
        """Run a chat completion. `messages` is a list of {"role","content"} dicts
        in OpenAI format. Returns the assistant's reply text."""
