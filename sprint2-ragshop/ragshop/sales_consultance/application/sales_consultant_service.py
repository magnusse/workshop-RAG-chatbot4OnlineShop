from ragshop.sales_consultance.domain.ports.llm_port import LLMPort


_SYSTEM_PROMPT = (
    "You are a friendly salesperson for household appliances. "
    "Answer the customer's questions concisely and politely."
)


class SalesConsultantService:
    """Application service: takes a question, asks the LLM, returns the answer.
    No history, no retrieval — that's what Sprints 3 and 4 will add."""

    def __init__(self, llm: LLMPort):
        self._llm = llm

    # Domain Story Step 3: Customer clarifies questions WITH Sales Assistant
    def ask(self, question_text: str) -> str:
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": question_text},
        ]
        # Domain Story Step 6: generate prompt AND send to LLM
        # Domain Story Step 7: LLM responds with Answer
        answer = self._llm.generate(messages)
        # Domain Story Step 8: Sales Assistant explains Details to Customer
        return answer
