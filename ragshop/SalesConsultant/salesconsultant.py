import json
import os
from dotenv import load_dotenv
load_dotenv()

#from ragshop.Retriever import retriever
from ragshop.Chatbot.CustomLLM import WPSCustomLLM
from ragshop.Retriever.retriever import IProductRetriever

#from ragshop.Retriever.retriever import load_vectorstore
from abc import ABC, abstractmethod

class ISalesConsultant(ABC):

    @abstractmethod
    def ask_qa_chain(self, prompt: str) -> str:
        """Stellt eine Frage und liefert die Antwort des LLM zurück."""
        pass

#Implementierung als Mock
class mock_salesconsultant(ISalesConsultant):

    def __init__(self, retriever:IProductRetriever):
        self.test = "www"

    def ask_qa_chain(self,prompt):
        result = "Great Question! We will be happy to answer your question personllay. Please call +49 0123 4567890"
        return result


#
# Die Implementierung mit einem LLM
#
class salesconsultant(ISalesConsultant):

    def __init__(self, retriever:IProductRetriever):
        self.retriever = retriever
        token = os.getenv("WEBUI_API_KEY")
        # TODO Integrate Abstraction
        self.llm = WPSCustomLLM(api_key=token)

    def ask_qa_chain(self,prompt):

# Wir ziehen die Ergebnisse raus und concatenieren sie zu einem Text
        context = self.retriever.retrievecontent(prompt,3)

# Nun erzeugen wir einen Prompt und fügen die Ergebnisse aus der Vektordatenbank hinzu
        llm_prompt = "You are a friendly salesperson for household appliances. Please answer the following question: \"" + prompt + "\" and use only the information from the following text passages for this purpose." + context
        print("LLM-Prompt = \n"+llm_prompt)
        result = self.llm.call(llm_prompt)

        response = json.loads(result) if isinstance(result, str) else result
        answer = response["choices"][0]["message"]["content"]

        return answer

#-------------------------------------------

def chat():
    print("🛍️ Welcome to the sales consultance chatbot. For exit type 'exit':\n")
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
        print("❌ Please set environment WEBUI_API_KEY.")
    else:
        token = os.getenv("WEBUI_API_KEY")
        chat()