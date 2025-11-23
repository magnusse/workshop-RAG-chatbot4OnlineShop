import json
import os

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
