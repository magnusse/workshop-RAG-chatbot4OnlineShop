class SalesConsultantService:
    """First incarnation: returns a single canned answer for every question.
    Just enough to wire a chat UI through to a Python function."""

    FALLBACK_ANSWER = (
        "Great question! We will be happy to answer your question personally. "
        "Please call +49 0123 4567890."
    )

    # Domain Story Step 3: Customer clarifies questions WITH Sales Assistant
    def ask(self, question_text: str) -> str:
        # Domain Story Step 8: Sales Assistant explains Details to Customer
        return self.FALLBACK_ANSWER
