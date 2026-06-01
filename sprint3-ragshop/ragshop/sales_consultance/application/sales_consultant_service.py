from typing import List

from ..domain.model.product_match import ProductMatch
from ..domain.ports.llm_port import LLMPort
from ..domain.ports.product_knowledge_port import ProductKnowledgePort
from .api import SalesConsultantApi


_SYSTEM_PROMPT = (
    "You are a friendly salesperson for household appliances. "
    "Answer the user's questions using only the product information provided "
    "in the user turn. If the information is insufficient, say so politely."
)

_RETRIEVAL_HITS = 3


class SalesConsultantService(SalesConsultantApi):
    """Sprint 3: every customer question triggers retrieval from the PKB and
    answers from the LLM using ONLY the retrieved context."""

    def __init__(self, llm: LLMPort, product_knowledge: ProductKnowledgePort):
        self._llm = llm
        self._product_knowledge = product_knowledge

    # Domain Story Step 3: Customer clarifies questions WITH Sales Assistant
    def ask(self, question_text: str) -> str:
        # Domain Story Step 5: select best content from Product Knowledge Base
        matches = self._product_knowledge.find_matches(question_text, _RETRIEVAL_HITS)

        context = self._render_matches_as_context(matches)
        user_turn = (
            f"Question: {question_text}\n\n"
            f"Use only the following product information to answer:\n{context}"
        )
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_turn},
        ]

        # Domain Story Step 6: generate prompt AND send to LLM
        # Domain Story Step 7: LLM responds with Answer
        answer = self._llm.generate(messages)
        # Domain Story Step 8: Sales Assistant explains Details to Customer
        return answer

    # Domain Story Step 2: PIM updates new/changed product INTO Product Knowledge Base
    def upsert_product(self, product: dict) -> None:
        self._product_knowledge.upsert_product(product)

    @staticmethod
    def _render_matches_as_context(matches: List[ProductMatch]) -> str:
        if not matches:
            return "(no matching products found)"
        return "\n\n".join(
            f"[{m.product_id}] {m.name}"
            + (f" ({m.category})" if m.category else "")
            + (f" - {m.price}" if m.price else "")
            + f"\n{m.description}"
            for m in matches
        )
