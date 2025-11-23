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




