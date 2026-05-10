import json
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

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

    SYSTEM_PROMPT = (
        "You are a friendly salesperson for household appliances. "
        "Answer the user's questions using only the product information provided "
        "in each user turn. If the information is insufficient, say so politely."
    )

    REWRITE_SYSTEM_PROMPT = (
        "Given the conversation history and a follow-up question, rewrite the "
        "follow-up into a standalone question that can be understood without the "
        "history. Keep it concise. Output ONLY the rewritten question, nothing else."
    )

    def __init__(self, retriever:IProductRetriever):
        self.retriever = retriever
        token = os.getenv("WEBUI_API_KEY")
        # TODO Integrate Abstraction
        self.llm = WPSCustomLLM(api_key=token)
        # Clean Q/A history (no retrieved doc chunks) — kept lean across turns.
        self.history: list[dict] = []

    def _extract_content(self, raw_result) -> str:
        response = json.loads(raw_result) if isinstance(raw_result, str) else raw_result
        return response["choices"][0]["message"]["content"]

    def _rewrite_query(self, prompt: str) -> str:
        """Turn a follow-up question into a standalone query for the retriever.--> history-aware retriever"""
        if not self.history:
            return prompt
        rewrite_messages = [
            {"role": "system", "content": self.REWRITE_SYSTEM_PROMPT},
            *self.history,
            {"role": "user", "content": f"Follow-up question: {prompt}\n\nStandalone question:"},
        ]
        rewritten = self._extract_content(self.llm.call(rewrite_messages)).strip()
        print(f"Rewritten query for retriever: {rewritten}")
        return rewritten

    def ask_qa_chain(self, prompt):
        standalone_query = self._rewrite_query(prompt)
        context = self.retriever.retrievecontent(standalone_query, 3)

        user_turn = (
            f"Question: {prompt}\n\n"
            f"Use only the following product information to answer:\n{context}"
        )
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            *self.history,
            {"role": "user", "content": user_turn},
        ]
        print("LLM messages =\n" + json.dumps(messages, ensure_ascii=False, indent=2))

        answer = self._extract_content(self.llm.call(messages))

        # Persist only the clean user question + answer, not the retrieved chunks,
        # so the history stays small over many turns.
        self.history.append({"role": "user", "content": prompt})
        self.history.append({"role": "assistant", "content": answer})

        return answer

    def reset_history(self) -> None:
        self.history = []
