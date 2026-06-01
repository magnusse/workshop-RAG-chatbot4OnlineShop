from typing import Callable, List, Optional

from ragshop.sales_consultance.domain.ports.llm_port import LLMPort


class FakeLLM(LLMPort):
    """Programmable fake LLM. Either return canned responses in order, or
    use a custom routing function based on the last user message."""

    def __init__(
        self,
        responses: Optional[List[str]] = None,
        router: Optional[Callable[[List[dict]], str]] = None,
    ):
        self._responses = list(responses or [])
        self._router = router
        self.calls: List[List[dict]] = []

    def generate(self, messages: List[dict]) -> str:
        self.calls.append(messages)
        if self._router is not None:
            return self._router(messages)
        if not self._responses:
            raise AssertionError("FakeLLM: no responses left")
        return self._responses.pop(0)
