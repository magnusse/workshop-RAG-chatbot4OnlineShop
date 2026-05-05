# RAG Chatbot für einen Online-Shop

Ein Workshop-Projekt, das einen KI-gestützten Verkaufsberater implementiert.
Nutzer können Produktfragen in natürlicher Sprache stellen; der Chatbot antwortet
auf Basis eines semantisch durchsuchten Produktkatalogs (Retrieval-Augmented Generation).

---

## Architektur

```
User
 └─► Gradio UI (UseInterface.py)
      └─► SalesConsultant.ask_qa_chain()
           ├─► Retriever  →  ChromaDB (Embeddings: all-MiniLM-L12-v2)
           └─► WPSCustomLLM  →  gpt.wps.de API (Mistral Medium)
```

---

## Projektstruktur

```
ragshop/
  Chatbot/CustomLLM.py                  # HTTP-Wrapper für gpt.wps.de
  Retriever/retriever.py                # ChromaDB-Anbindung + Embedding
  Retriever/preprocessing.py            # Datenaufbereitung & Vektorisierung
  SalesConsultant/salesconsultant.py    # RAG-Orchestrierung
  SalesConsultant/UseInterface.py       # Gradio-Einstiegspunkt

data/raw/products.json                  # Produktkatalog (~20 Produkte)
vectorstore/chromadb/                   # Persistenter Vektorspeicher (vorbelegt)

sprint1-ragshop/ … sprint4-ragshop/    # Schrittweise Workshop-Stufen
tests/                                  # Teststubs
```

---

## Voraussetzungen

- Python 3.11+
- Zugang zur WPS-LLM-API (`gpt.wps.de`) mit einem gültigen API-Key

---

## API-Key einrichten

Der Chatbot nutzt den LLM-Endpoint `https://gpt.wps.de/api/chat/completions`.
Der API-Key wird als Umgebungsvariable `WEBUI_API_KEY` erwartet.

```bash
# .env Datei im Projektverzeichnis anlegen:
echo "WEBUI_API_KEY=dein_token_hier" > .env
```

> Die `.env`-Datei niemals ins Repository einchecken.

---

## Installation & Start

```bash
# 1. Virtuelle Umgebung aktivieren
source .venv/bin/activate

# 2. Abhängigkeiten installieren
pip install -r requirements.txt

# 3. (Optional) Vektordatenbank neu aufbauen
python -m ragshop.Retriever.preprocessing rebuild

# 4. Chatbot starten
python -m ragshop.SalesConsultant.UseInterface
```

Die Gradio-Oberfläche ist anschließend unter **http://localhost:7860** erreichbar.

---

## Verwendete Modelle

| Zweck      | Modell                           |
|------------|----------------------------------|
| LLM        | Mistral Medium Latest (WPS API)  |
| Embeddings | all-MiniLM-L12-v2 (HuggingFace)  |

---

## Workshop-Stufen

Das Projekt zeigt schrittweise, wie eine RAG-Pipeline aufgebaut wird:

| Sprint   | Inhalt                                              |
|----------|-----------------------------------------------------|
| sprint1  | Nur Mock-Implementierungen (kein LLM, kein Retriever) |
| sprint2  | Echtes LLM, Mock-Retriever                          |
| sprint3  | Wie sprint2, deutschsprachige Prompts               |
| sprint4  | Vollständige RAG-Pipeline (LLM + ChromaDB)          |

---

## Tests ausführen

```bash
pytest tests/
```
