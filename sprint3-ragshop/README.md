# Sprint 3 — Chatbot über RAG

Der Chatbot kennt jetzt unser Sortiment: Beim Start werden die Produkte aus `data/raw/products.json` in eine Vektordatenbank gepusht, und jede Kundenfrage wird mit den relevantesten Treffern aus dieser Datenbank an das LLM gegeben.

## Abgedeckte Schritte der Domain Story

Zusätzlich zu den Sprints 1–2:

- **Schritt 2** — *PIM updates new/changed product INTO Product Knowledge Base*
- **Schritt 5** — *Sales Assistant selects best content from Product Knowledge Base*

## Bausteine — und warum jetzt (Last Responsible Moment)

| Baustein | Begründung |
|---|---|
| `product_information_management/` (PIM-Kontext) | Erstmals haben wir Stammdaten zu verwalten — Produkte. Das gehört nicht zu Sales Consultance, also ein eigener Bounded Context. |
| `Product`, `JsonProductRepository`, `bootstrap.py` | PIM braucht ein Master-Datenmodell und eine Lese-Quelle. JSON-Datei reicht aus — kein DB-Server, keine Schreibpfade (Sprint 5). |
| `ProductKnowledgeSink` (PIM-Outbound-Port) | PIM muss die PKB befüllen können, soll aber Sales Consultance nicht kennen. Outbound-Interface bricht die Abhängigkeit. |
| `ProductKnowledgePort` (Sales-Consultance-Outbound-Port) | Wir wollen die PKB austauschen können (Fakes für Tests, evtl. anderes Backend). |
| `ChromaProductKnowledgeAdapter` | ChromaDB ist eine pragmatische Wahl: persistent, embedded, keine Infra. Chunking-Logik lebt hier, weil "wie indexieren" ein Retrieval-Detail ist. |
| `ProductMatch` (Value Object) | Erstes Domain-Modell-Objekt: ein Treffer aus der PKB hat Struktur (id, name, beschreibung), nicht nur Text. Plain `str` würde reichen, aber die Application Logic braucht zumindest `name`, um sie an das LLM weiterzugeben. |
| `SalesConsultantApi` (Inbound-Port) | Erstmals haben wir zwei Konsumenten von Sales Consultance: Gradio (ask) und PIM (upsert). Ein expliziter Eingangs-Vertrag macht das sauber. |
| `PimSinkAdapter` | Implementiert PIMs `ProductKnowledgeSink`, ruft `SalesConsultantApi.upsert_product`. |

## Was bewusst noch fehlt — und warum

- **Keine Konversations-Historie.** Folgefragen wie *„Und wie viel kostet der?"* funktionieren noch nicht. Kommt in Sprint 4.
- **Kein `delete_product` / kein Update-Pfad.** Sprint 3 macht nur Bootstrap (idempotent: jeder Start synchronisiert komplett). Laufzeit-Updates und Löschungen kommen in Sprint 5.
- **Keine Filter beim Retrieval.** Wir holen einfach die 3 ähnlichsten Treffer — keine Berücksichtigung von Aktualität (`upddate`) oder Lösch-Flags (`delflag`). Kommt in Sprint 5.
- **Kein Intent-Routing.** Jede Frage wird wie eine Produktfrage behandelt. Kommt in Sprint 6.
- **Keine `Question`-/`Answer`-VOs.** Plain `str`. Die kommen erst, wenn die Conversation in Sprint 4 sie strukturell braucht.

## Starten

```bash
cd sprint3-ragshop
pip install -r requirements.txt
python -m ragshop.sales_consultance.interfaces.gradio_chat_interface
```

Beim ersten Lauf wird das Sentence-Transformer-Modell heruntergeladen (~120 MB) und die Vektordatenbank in `vectorstore/chromadb/` erzeugt.

## Tests

```bash
cd sprint3-ragshop
pytest tests/ -q
```
