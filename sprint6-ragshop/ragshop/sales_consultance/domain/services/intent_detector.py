from typing import List

from ..model.intent import Intent
from ..ports.llm_port import LLMPort


_SYSTEM_PROMPT = (
    "You classify a customer's latest message in a sales consultation chat. "
    "Choose exactly one of: PRODUCT_INFO, OFFER_REQUEST, SMALLTALK. "
    "PRODUCT_INFO = customer asks about products, features, prices, recommendations. "
    "OFFER_REQUEST = customer asks to receive/create an offer or to order. "
    "SMALLTALK = greetings, thanks, anything else. "
    "Reply with ONLY the label, nothing else."
)


class IntentDetector:
    """Domain Story Step 4: Sales Assistant detects Intent (Product Info / Offer Request).

    Implemented as an LLM-backed classifier via the LLMPort, so the domain
    has no direct dependency on any LLM provider.
    """

    def __init__(self, llm: LLMPort):
        self._llm = llm

    def detect(self, question_text: str, history: List[dict]) -> Intent:
        # Domain Story Step 4: Sales Assistant detects Intent
        messages: List[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]
        messages.extend(history)
        messages.append(
            {"role": "user", "content": f"Classify this message: {question_text}"}
        )
        raw = self._llm.generate(messages)
        return Intent.parse(raw)
