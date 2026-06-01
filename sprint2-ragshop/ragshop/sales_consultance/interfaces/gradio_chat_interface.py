from dotenv import find_dotenv, load_dotenv
import gradio as gr

from ragshop.composition_root import build_sales_consultant


def launch() -> None:
    load_dotenv(find_dotenv())
    service = build_sales_consultant()

    # Domain Story Step 3: Customer clarifies questions WITH Sales Assistant
    def respond(message: str, _history):
        try:
            return service.ask(message)
        except Exception as e:  # noqa: BLE001 — surface errors in the chat UI
            print(f"ERROR in Chatbot: {e}")
            return f"Fatal error in Chatbot: {e}"

    gr.ChatInterface(fn=respond).launch()


if __name__ == "__main__":
    launch()
