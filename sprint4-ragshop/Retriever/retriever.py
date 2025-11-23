from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from ragshop.Retriever.preprocessing import COLLECTION_NAME
from ragshop.Retriever.preprocessing import EMBEDDING_MODEL_NAME
from abc import ABC, abstractmethod

# Einstellungen
DB_DIR = "../vectorstore/chromadb"



class IProductRetriever(ABC):
    @abstractmethod
    def retrievecontent(self, prompt, hits):
        """Stellt eine Frage und liefert die Antwort des LLM zurück."""
        pass

class mock_productretriever(IProductRetriever):

    def __init__(self):
        self.text = "Hallo"

    def retrievecontent(self, prompt, hits):

        # Wir ziehen die Ergebnisse raus und concatenieren sie zu einem Text

        context = "EcoClean 2000 mit HEPA Filter"

        return context


class productretriever(IProductRetriever):

    def __init__(self):
        self.__client = PersistentClient(DB_DIR)
        self.__embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)
        self.__collection = self.__client.get_collection(COLLECTION_NAME)
        self.__embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    def retrievecontent(self, prompt, hits):
        # Ganz wichtig: Aus der Frage des Users muss mit dem gleichen Embedding Modell wie bei der VektorDB die Frage embedded werden
        embedding = self.__embedding_model.encode(prompt)

        # Jetzt damit die Anfrage in der VektorDB stellen. Bei CromaDB könnte man hier auch mehrere Query embeddings übergeben
        results = self.__collection.query(
            query_embeddings=[embedding],  # Muss eine Liste sein!
            n_results=hits
        )

        # Wir ziehen die Ergebnisse raus und concatenieren sie zu einem Text
        retrieved_texts = results["documents"][0]
        context = "\n\n".join(retrieved_texts)

        return context


