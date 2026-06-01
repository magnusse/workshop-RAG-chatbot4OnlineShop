import gradio as gr

from ragshop.sales_consultance.application.sales_consultant_service import (
    SalesConsultantService,
)


def launch() -> None:
    service = SalesConsultantService()

    # Domain Story Step 3: Customer clarifies questions WITH Sales Assistant
    def respond(message: str, _history):
        return service.ask(message)

    gr.ChatInterface(fn=respond).launch()


if __name__ == "__main__":
    launch()
