
import gradio as gr
import json
from ragshop.SalesConsultant.salesconsultant import salesconsultant  # Domain Logic
from ragshop.SalesConsultant.salesconsultant import mock_salesconsultant

# Initialisiere den Chatbot (kann auch lazy init sein)
myChatbot = salesconsultant()
# myChatbot = mock_salesconsultant()

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

