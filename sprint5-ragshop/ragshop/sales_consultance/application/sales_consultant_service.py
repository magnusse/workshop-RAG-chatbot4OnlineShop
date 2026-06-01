from typing import List

from ..domain.model.conversation import ConversationId
from ..domain.model.customer import CustomerId
from ..domain.model.product_match import ProductMatch
from ..domain.model.question import Answer, Question
from ..domain.ports.conversation_repository import ConversationRepository
from ..domain.ports.llm_port import LLMPort
from ..domain.ports.product_knowledge_port import ProductKnowledgePort
from ..domain.services.query_rewriter import QueryRewriter
from .api import SalesConsultantApi


_SYSTEM_PROMPT = (
    "You are a friendly salesperson for household appliances. "
    "Answer the user's questions using only the product information provided "
    "in each user turn. If the information is insufficient, say so politely."
)

_RETRIEVAL_HITS = 3


class SalesConsultantService(SalesConsultantApi):
    """Sprint 4: conversation-aware. Each ask() loads the prior turns, rewrites
    follow-up questions into standalone queries (history-aware retrieval),
    asks the LLM with the full history, and persists the new turn."""

    def __init__(
        self,
        llm: LLMPort,
        product_knowledge: ProductKnowledgePort,
        conversations: ConversationRepository,
    ):
        self._llm = llm
        self._product_knowledge = product_knowledge
        self._conversations = conversations
        self._query_rewriter = QueryRewriter(llm)

    def start_conversation(self, customer_id: CustomerId) -> ConversationId:
        return self._conversations.create(customer_id).id

    # Domain Story Step 3: Customer clarifies questions WITH Sales Assistant
    def ask(self, conversation_id: ConversationId, question_text: str) -> str:
        question = Question(text=question_text)
        conversation = self._conversations.find(conversation_id)
        if conversation is None:
            raise LookupError(f"Conversation not found: {conversation_id.value}")

        history = conversation.as_history_messages()

        standalone_query = self._query_rewriter.rewrite(question.text, history)

        # Domain Story Step 5: select best content from Product Knowledge Base
        matches = self._product_knowledge.find_matches(standalone_query, _RETRIEVAL_HITS)

        context = self._render_matches_as_context(matches)
        user_turn = (
            f"Question: {question.text}\n\n"
            f"Use only the following product information to answer:\n{context}"
        )
        messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_turn})

        # Domain Story Step 6: generate prompt AND send to LLM
        # Domain Story Step 7: LLM responds with Answer
        answer_text = self._llm.generate(messages)
        answer = Answer(text=answer_text)

        # Domain Story Step 8: Sales Assistant explains Details (records turn)
        conversation.record_turn(question, answer, matches)
        self._conversations.save(conversation)
        return answer.text

    # Domain Story Step 2: PIM updates new/changed product INTO Product Knowledge Base
    def upsert_product(self, product: dict) -> None:
        self._product_knowledge.upsert_product(product)

    def delete_product(self, product_id: str) -> None:
        self._product_knowledge.delete_product(product_id)

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
