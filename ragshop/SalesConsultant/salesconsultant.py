import json
import os

#from ragshop.Retriever import retriever
from ragshop.Chatbot.CustomLLM import WPSCustomLLM
from ragshop.Retriever.retriever import productretriever
#from ragshop.Retriever.retriever import load_vectorstore


class salesconsultant:

    def __init__(self):
        self.retriever = productretriever()
        # TODO: Dies nur einmal setzen und von außen verfügbar machen
        token = os.getenv("WEBUI_API_KEY")
        self.llm = WPSCustomLLM(api_key=token)

    def ask_qa_chain(self,prompt):

# Wir ziehen die Ergebnisse raus und concatenieren sie zu einem Text
        context = self.retriever.retrievecontent(prompt,3)

# Nun erzeugen wir einen Prompt und fügen die Ergebnisse aus der Vektordatenbank hinzu
        llm_prompt = "Du bist ein freundlicher Produktverkäufer für Haushaltsgeräte. Bitte beantworte die folgende Frage: \"" + prompt + "\" und benutze dafür ausschließlich die Informationen aus den folgenden Textstücken:" + context
        print("LLM-Prompt = \n"+llm_prompt)
        result = self.llm.call(llm_prompt)

        return result

#-------------------------------------------

def chat():
    print("🛍️ Willkommen beim Produkt-Chatbot. Stelle deine Frage oder tippe 'exit':\n")
    # collection = load_vectorstore()
    myChatbot = salesconsultant()

    while True:
        query = input("Du: ")
        if query.lower() in ("exit", "quit"):
            break
        result = myChatbot.ask_qa_chain(query)
        response = json.loads(result)

        print("\nAntwort:")
        print(response["choices"][0]["message"]["content"])
        # print(result)

        print("\n---\n")

if __name__ == "__main__":
    if not os.getenv("WEBUI_API_KEY"):
        print("❌ Bitte setze die Umgebungsvariable WEBUI_API_KEY.")
    else:
        token = os.getenv("WEBUI_API_KEY")
        chat()