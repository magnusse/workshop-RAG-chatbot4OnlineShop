# RAG Chatbot für einen Online-Shop

Ein Workshop-Projekt, das einen KI-gestützten Verkaufsberater implementiert.
Nutzer können Produktfragen in natürlicher Sprache stellen; der Chatbot antwortet
auf Basis eines semantisch durchsuchten Produktkatalogs (Retrieval-Augmented Generation).

---

## Architektur

```
User
 └─► Gradio UI (gradio_chat_interface.py)
      └─► SalesConsultantService.ask()
           ├─► ChromaProductKnowledgeAdapter  →  ChromaDB (Embeddings: all-MiniLM-L12-v2)
           └─► WpsLLMAdapter  →  gpt.wps.de API (Mistral Medium)
```

Die Anwendung folgt einer hexagonalen Architektur mit drei Bounded Contexts:
- **Sales Consultance** – Kernlogik des Verkaufsberaters
- **Product Information Management (PIM)** – Produktkatalog und Vektorisierung
- **Order Management** – Angebotserstellung und -verwaltung

---

## Projektstruktur

```
ragshop/
  composition_root.py                                      # Verdrahtung aller Adapter (einzige Stelle mit konkreten Abhängigkeiten)
  sales_consultance/
    application/sales_consultant_service.py                # RAG-Orchestrierung
    domain/                                                # Domänenmodell & Ports
    infrastructure/wps_llm_adapter.py                      # HTTP-Wrapper für gpt.wps.de
    infrastructure/chroma_product_knowledge_adapter.py     # ChromaDB-Anbindung
    interfaces/gradio_chat_interface.py                    # Gradio-Einstiegspunkt
  product_information_management/
    product_catalog_service.py                             # Produktkatalog-Verwaltung
    bootstrap.py                                           # Initiales Befüllen der Vektordatenbank
  order_management/
    order_service.py                                       # Angebotserstellung

data/raw/products.json                  # Produktkatalog (~20 Produkte)
vectorstore/chromadb/                   # Persistenter Vektorspeicher (vorbelegt)

sprint1-ragshop/ … sprint6-ragshop/    # Schrittweise Workshop-Stufen
tests/                                  # Teststubs
```

---

## Voraussetzungen

- Python 3.10+
- Zugang zur WPS-LLM-API (`gpt.wps.de`) mit einem gültigen API-Key

---

## API-Key einrichten

Der Chatbot nutzt den LLM-Endpoint `https://gpt.wps.de/api/chat/completions`.
Der API-Key wird nach folgender Priorität aufgelöst:

1. **OS-Keyring** (macOS Keychain / Windows Credential Manager / Linux Secret Service) – wird beim ersten erfolgreichen Login automatisch gespeichert
2. **Umgebungsvariable** `WEBUI_API_KEY` – wird beim ersten Lesen in den Keyring übertragen
3. **Interaktive Eingabe** beim Start (nur im Terminal) – wird ebenfalls im Keyring gespeichert

```bash
# Einmalig setzen (wird danach im Keyring gespeichert):
export WEBUI_API_KEY=dein_token_hier

# Keyring-Eintrag zurücksetzen (z. B. nach Key-Rotation):
WEBUI_API_KEY_RESET=1 python -m ragshop.sales_consultance.interfaces.gradio_chat_interface
```

---

## Installation & Start

```bash
# 1. Python 3.10 installieren (einmalig, falls nicht vorhanden)
pyenv install 3.10
pyenv local 3.10

# 2. Virtuelle Umgebung anlegen und aktivieren
python3.10 -m venv .venv
source .venv/bin/activate

# 3. Abhängigkeiten installieren
pip install -r requirements.txt

# 4. Chatbot starten
python -m ragshop.sales_consultance.interfaces.gradio_chat_interface
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

| Sprint   | Inhalt                                                                        |
|----------|-------------------------------------------------------------------------------|
| sprint1  | Chatbot-Basis: feste Antwort, Frontend und Backend verbunden                  |
| sprint2  | Echtes LLM, noch kein RAG und keine Gesprächshistorie                         |
| sprint3  | Vollständige RAG-Pipeline: Produkte aus ChromaDB als Kontext                  |
| sprint4  | Gesprächshistorie: history-aware Retrieval mit Folgefragen                    |
| sprint5  | Produktaktualisierungen zur Laufzeit, Retrieval-Filter für gelöschte Einträge |
| sprint6  | Intent-Erkennung (Produktfrage / Smalltalk / Angebotswunsch) und Order Management |

---

## Tests ausführen

```bash
pytest tests/
```
