from dotenv import find_dotenv, load_dotenv
import gradio as gr

from ragshop.composition_root import build_application
from ragshop.sales_consultance.domain.model.customer import CustomerId


def launch() -> None:
    load_dotenv(find_dotenv())
    # bootstrap=True is idempotent: it upserts every product from products.json
    # into the PKB on every launch, so the store always reflects the master data.
    app = build_application(bootstrap_pkb=True)

    # One conversation per process — fine for the workshop demo. A multi-user
    # deployment would use gr.State to keep a conversation per session.
    conversation_id = app.sales_consultant.start_conversation(
        CustomerId(value="guest")
    )

    # Domain Story Step 3: Customer clarifies questions WITH Sales Assistant
    def respond(message: str, _history):
        try:
            return app.sales_consultant.ask(conversation_id, message)
        except Exception as e:  # noqa: BLE001 — surface errors in the chat UI
            print(f"ERROR in Chatbot: {e}")
            return f"Fatal error in Chatbot: {e}"

    gr.ChatInterface(fn=respond).launch()


if __name__ == "__main__":
    launch()
