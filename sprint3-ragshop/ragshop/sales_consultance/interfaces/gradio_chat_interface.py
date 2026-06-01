from dotenv import find_dotenv, load_dotenv
import gradio as gr

from ragshop.composition_root import build_application


def launch() -> None:
    load_dotenv(find_dotenv())
    # Idempotent: re-syncs products.json -> PKB on every startup.
    app = build_application(bootstrap_pkb=True)

    # Domain Story Step 3: Customer clarifies questions WITH Sales Assistant
    def respond(message: str, _history):
        try:
            return app.sales_consultant.ask(message)
        except Exception as e:  # noqa: BLE001
            print(f"ERROR in Chatbot: {e}")
            return f"Fatal error in Chatbot: {e}"

    gr.ChatInterface(fn=respond).launch()


if __name__ == "__main__":
    launch()
