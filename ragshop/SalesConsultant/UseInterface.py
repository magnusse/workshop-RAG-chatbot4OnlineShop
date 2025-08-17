
import gradio as gr
import json
from ragshop.SalesConsultant.salesconsultant import salesconsultant  # Domain Logic

# Initialisiere den Chatbot (kann auch lazy init sein)
myChatbot = salesconsultant()

# Call Back für das Frontend
def respond(message, history):
    try:
        result = myChatbot.ask_qa_chain(message)
        response = json.loads(result) if isinstance(result, str) else result
        answer = response["choices"][0]["message"]["content"]
        return answer
    except Exception as e:
        return "Fehler in Domain Logic"

gr.ChatInterface(
        fn=respond,
        type="messages"
).launch()

