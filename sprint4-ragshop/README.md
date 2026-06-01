# Sprint 4 — Chatbot Historie

Der Chatbot kann jetzt eine echte Konversation führen: Folgefragen wie *„Und in weiß?"* werden im Kontext der vorherigen Turns verstanden, weil sie vor dem Retrieval in eigenständige Suchanfragen umformuliert werden (history-aware retrieval).

## Abgedeckte Schritte der Domain Story

Sprints 1–3 wiederholbar gemacht — die Schritte 3 bis 8 laufen jetzt im Loop:

- **Schritt 3** — *Customer clarifies questions* (jetzt mehrfach pro Session)
- **Schritt 5** — *select best content* (mit umgeschriebener Query)
- **Schritt 6** — *generate prompt AND send* (mit voller History im Prompt)
- **Schritt 7** — *LLM responds with Answer*
- **Schritt 8** — *Sales Assistant explains Details* (Antwort wird in der Conversation persistiert)

## Bausteine — und warum jetzt (Last Responsible Moment)

| Baustein | Begründung |
|---|---|
| `Conversation` (Aggregate Root) | Erstmals haben wir Zustand über Aufrufe hinweg. Eine Aggregate-Root bündelt die Turns und schützt die Invariante „Reihenfolge bleibt erhalten". |
| `Turn` (Entity) | Eine Konversation besteht aus Q/A-Turns mit Metadaten (matches, Zeitstempel). Eine Liste von Tuples wäre zu schwach typisiert. |
| `Question`, `Answer` (Value Objects) | Frage/Antwort werden jetzt strukturiert gespeichert. `Question` validiert leere Eingaben — die Aggregate-Invariante hat ihre erste echte Regel. |
| `Customer`, `CustomerId` (VO) | Konversation gehört zu einem Kunden. Vorerst nur `CustomerId` als String-Wrapper, aber der Platz für künftige Customer-Attribute ist da. |
| `ConversationRepository` (Outbound-Port) | Application Service braucht Lookup nach `ConversationId`. Port erlaubt Test-Fakes und spätere Persistierung außerhalb des Prozesses. |
| `InMemoryConversationRepository` | Reichweite für Workshop-Demos. Datei-/DB-Persistierung kommt erst bei multi-user. |
| `QueryRewriter` (Domain Service) | Pure Domain-Logik: aus History + Frage eine eigenständige Suchanfrage machen. Lebt in der Domain, weil es Konversationsregeln kapselt — die Implementierung delegiert technisch ans LLM via `LLMPort`. |

## Was bewusst noch fehlt — und warum

- **Single-User-Demo.** Eine `ConversationId` pro Prozess. Mehrere parallele Benutzer mit eigenen Sessions würden Gradio `gr.State` brauchen — zusätzliche Komplexität, ohne Lernwert für das aktuelle Thema.
- **Kein Product-Update-Pfad.** Die PKB wird weiterhin nur beim Bootstrap befüllt. Kommt in Sprint 5.
- **Keine Retrieval-Filter (`delflag`, `upddate`).** Wir holen weiterhin die Top-K-Treffer ohne Sortierregeln. Kommt in Sprint 5.
- **Kein Intent-Routing.** Smalltalk und Offer-Wünsche werden noch wie Produktfragen behandelt. Kommt in Sprint 6.

## Starten

```bash
cd sprint4-ragshop
pip install -r requirements.txt
python -m ragshop.sales_consultance.interfaces.gradio_chat_interface
```

## Tests

```bash
cd sprint4-ragshop
pytest tests/ -q
```
