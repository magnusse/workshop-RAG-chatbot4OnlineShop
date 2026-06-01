from typing import List

from ..domain.model.conversation import ConversationId
from ..domain.model.customer import CustomerId
from ..domain.model.intent import Intent
from ..domain.model.offer import Offer, OfferLine
from ..domain.model.product_match import ProductMatch
from ..domain.model.question import Answer, Question
from ..domain.ports.conversation_repository import ConversationRepository
from ..domain.ports.llm_port import LLMPort
from ..domain.ports.order_management_port import OrderManagementPort
from ..domain.ports.product_knowledge_port import ProductKnowledgePort
from ..domain.services.intent_detector import IntentDetector
from ..domain.services.query_rewriter import QueryRewriter
from .api import SalesConsultantApi


_SYSTEM_PROMPT = (
    "You are a friendly salesperson for household appliances. "
    "Answer the user's questions using only the product information provided "
    "in each user turn. If the information is insufficient, say so politely."
)

_OFFER_EXTRACTION_PROMPT = (
    "From the conversation context and the customer's latest message, list the "
    "product IDs the customer wants in their offer. Return ONLY a comma-separated "
    "list of product IDs from the candidates below. If none of the candidates fit, "
    "return the single word NONE."
)

_RETRIEVAL_HITS = 3


class SalesConsultantService(SalesConsultantApi):
    """Application service orchestrating the sales consultation use case.
    Lives in the application layer — depends on domain ports, never on
    concrete infrastructure."""

    def __init__(
        self,
        llm: LLMPort,
        product_knowledge: ProductKnowledgePort,
        order_management: OrderManagementPort,
        conversations: ConversationRepository,
    ):
        self._llm = llm
        self._product_knowledge = product_knowledge
        self._order_management = order_management
        self._conversations = conversations
        self._intent_detector = IntentDetector(llm)
        self._query_rewriter = QueryRewriter(llm)

    def start_conversation(self, customer_id: CustomerId) -> ConversationId:
        conversation = self._conversations.create(customer_id)
        return conversation.id

    # Domain Story Step 3: Customer clarifies questions WITH Sales Assistant
    def ask(self, conversation_id: ConversationId, question_text: str) -> str:
        question = Question(text=question_text)
        conversation = self._conversations.find(conversation_id)
        if conversation is None:
            raise LookupError(f"Conversation not found: {conversation_id.value}")

        history = conversation.as_history_messages()

        # Domain Story Step 4: Sales Assistant detects Intent
        intent = self._intent_detector.detect(question.text, history)

        if intent == Intent.OFFER_REQUEST:
            answer_text, matches = self._handle_offer_request(conversation, question)
        elif intent == Intent.PRODUCT_INFO:
            answer_text, matches = self._handle_product_info(question, history)
        else:
            answer_text, matches = self._handle_smalltalk(question, history)

        answer = Answer(text=answer_text)
        # Domain Story Step 8: Sales Assistant explains Details (records turn)
        conversation.record_turn(question, answer, intent, matches)
        self._conversations.save(conversation)
        return answer.text

    # Domain Story Step 2: PIM updates new/changed product INTO Product Knowledge Base
    def upsert_product(self, product: dict) -> None:
        self._product_knowledge.upsert_product(product)

    def delete_product(self, product_id: str) -> None:
        self._product_knowledge.delete_product(product_id)

    # --- internal handlers -------------------------------------------------

    def _handle_product_info(
        self, question: Question, history: List[dict]
    ) -> tuple[str, List[ProductMatch]]:
        standalone_query = self._query_rewriter.rewrite(question.text, history)

        # Domain Story Step 5: select best content from Product Knowledge Base
        matches = self._product_knowledge.find_matches(standalone_query, _RETRIEVAL_HITS)

        context = self._render_matches_as_context(matches)
        user_turn = (
            f"Question: {question.text}\n\n"
            f"Use only the following product information to answer:\n{context}"
        )

        # Domain Story Step 6: generate prompt AND send to LLM
        messages: List[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_turn})

        # Domain Story Step 7: LLM responds with Answer
        answer_text = self._llm.generate(messages)
        return answer_text, matches

    def _handle_smalltalk(
        self, question: Question, history: List[dict]
    ) -> tuple[str, List[ProductMatch]]:
        # No retrieval: skip the PKB for greetings/thanks/etc.
        messages: List[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]
        messages.extend(history)
        messages.append({"role": "user", "content": question.text})
        return self._llm.generate(messages), []

    # Domain Story Step 9: Customer requests offer
    def _handle_offer_request(
        self, conversation, question: Question
    ) -> tuple[str, List[ProductMatch]]:
        candidates = conversation.recent_matches(limit=10)
        selected_ids = self._extract_offer_product_ids(question.text, candidates)
        chosen = [m for m in candidates if m.product_id in selected_ids]

        if not chosen:
            answer = (
                "Gerne erstelle ich ein Angebot. Welches Produkt aus unserem Sortiment "
                "moechten Sie konkret im Angebot haben?"
            )
            return answer, []

        offer = Offer(
            customer_id=conversation.customer_id,
            lines=[
                OfferLine(
                    product_id=m.product_id, name=m.name, price=m.price, quantity=1
                )
                for m in chosen
            ],
        )

        # Domain Story Step 10: Sales Assistant creates & sends Offer Request
        receipt = self._order_management.submit_offer_request(offer)

        product_names = ", ".join(line.name for line in offer.lines)
        answer = (
            f"Ihr Angebot fuer {product_names} wurde erstellt "
            f"(Angebotsnummer: {receipt.offer_id}). Sie koennen es im Warenkorb einsehen."
        )
        return answer, chosen

    def _extract_offer_product_ids(
        self, question_text: str, candidates: List[ProductMatch]
    ) -> set[str]:
        if not candidates:
            return set()
        candidate_block = "\n".join(
            f"- {m.product_id}: {m.name}" for m in candidates
        )
        messages = [
            {"role": "system", "content": _OFFER_EXTRACTION_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Candidates:\n{candidate_block}\n\n"
                    f"Customer message: {question_text}"
                ),
            },
        ]
        raw = self._llm.generate(messages).strip()
        if raw.upper() == "NONE":
            return set()
        valid_ids = {m.product_id for m in candidates}
        return {token.strip() for token in raw.split(",") if token.strip() in valid_ids}

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
