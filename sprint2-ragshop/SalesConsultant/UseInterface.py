
import gradio as gr
from ragshop.SalesConsultant.salesconsultant import salesconsultant  # Domain Logic
from ragshop.Retriever.retriever import mock_productretriever

# Initialisiere Retriever
retriever = mock_productretriever()

# Initialisiere den Chatbot
myChatbot = salesconsultant(retriever)

# Call Back für das Frontend
def respond(message, history):
    try:
        result = myChatbot.ask_qa_chain(message)
        return result

    except Exception as e:
        return "Fatal error in Chatbot"

gr.ChatInterface(
        fn=respond,
        type="messages"
).launch()

