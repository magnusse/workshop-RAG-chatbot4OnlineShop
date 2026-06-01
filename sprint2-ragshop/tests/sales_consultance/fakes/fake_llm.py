from typing import List, Optional

from ragshop.sales_consultance.domain.ports.llm_port import LLMPort


class FakeLLM(LLMPort):
    """Returns canned responses in order. Records every call for assertions."""

    def __init__(self, responses: Optional[List[str]] = None):
        self._responses = list(responses or [])
        self.calls: List[List[dict]] = []

    def generate(self, messages: List[dict]) -> str:
        self.calls.append(messages)
        if not self._responses:
            raise AssertionError("FakeLLM: no responses left")
        return self._responses.pop(0)
