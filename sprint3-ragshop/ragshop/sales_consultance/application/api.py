from abc import ABC, abstractmethod


class SalesConsultantApi(ABC):
    """Inbound (driving) port. Introduced in Sprint 3 because Sales Consultance
    now has two distinct callers: the Gradio UI (ask) and the PIM context
    (upsert_product). An explicit interface keeps both decoupled from the
    concrete service implementation."""

    # Domain Story Step 3: Customer clarifies questions WITH Sales Assistant
    @abstractmethod
    def ask(self, question_text: str) -> str:
        ...

    # Domain Story Step 2: PIM updates new/changed product INTO Product Knowledge Base
    @abstractmethod
    def upsert_product(self, product: dict) -> None:
        ...
