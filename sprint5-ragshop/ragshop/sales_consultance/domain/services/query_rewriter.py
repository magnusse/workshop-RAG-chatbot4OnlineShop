from typing import List

from ..ports.llm_port import LLMPort


_SYSTEM_PROMPT = (
    "Given the conversation history and a follow-up question, rewrite the "
    "follow-up into a standalone question that can be understood without the "
    "history. Keep it concise. Output ONLY the rewritten question, nothing else."
)


class QueryRewriter:
    """Rewrites a follow-up question into a standalone query so the
    Product Knowledge Base receives a self-contained search input
    (history-aware retrieval)."""

    def __init__(self, llm: LLMPort):
        self._llm = llm

    def rewrite(self, question_text: str, history: List[dict]) -> str:
        if not history:
            return question_text
        messages: List[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]
        messages.extend(history)
        messages.append(
            {
                "role": "user",
                "content": f"Follow-up question: {question_text}\n\nStandalone question:",
            }
        )
        return self._llm.generate(messages).strip()
