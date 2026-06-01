# Sprint 1 — Chatbot Basis

Erster lauffähiger Schritt: ein Chatbot, der jede Frage des Kunden mit einer fixen Antwort beantwortet. Damit ist die Brücke zwischen Frontend und Backend geschlossen — alles weitere wächst von hier aus.

## Abgedeckte Schritte der Domain Story

- **Schritt 3** — *Customer clarifies questions WITH Sales Assistant*
- **Schritt 8** — *Sales Assistant explains Details to Customer*

## Bausteine — und warum jetzt (Last Responsible Moment)

| Baustein | Begründung |
|---|---|
| `SalesConsultantService.ask(...)` | Erste Verantwortlichkeit überhaupt: nimmt eine Frage entgegen, liefert eine Antwort. Eine einzige Methode reicht. |
| `gradio_chat_interface.py` | Wir brauchen einen Weg, die Antwort sichtbar zu machen. Gradio kostet keine Architekturentscheidung — eine Funktion `(message, history) -> str` genügt. |

## Was bewusst noch fehlt — und warum

- **Keine LLM-Anbindung.** Wir haben noch nicht entschieden, ob wir überhaupt ein LLM benötigen. Vielleicht wäre eine Lookup-Tabelle ausreichend. Diese Entscheidung wird in Sprint 2 getroffen.
- **Keine Ports oder Adapter.** Es gibt nichts zu substituieren — keine Tests gegen Fakes, kein zweites Backend. Ein `LLMPort`-Interface wäre jetzt reine Spekulation.
- **Keine Domain-Modelle (`Question`/`Answer` als VO).** `str` reicht aus. VOs werden eingeführt, sobald sie etwas kapseln oder validieren müssen.
- **Kein `composition_root`.** Der Service hat keine Dependencies; es gibt nichts zu verkabeln.
- **Keine Conversation/History.** Jede Frage wird isoliert beantwortet. History kommt in Sprint 4.

## Starten

```bash
cd sprint1-ragshop
pip install -r requirements.txt
python -m ragshop.sales_consultance.interfaces.gradio_chat_interface
```

## Tests

```bash
cd sprint1-ragshop
pytest tests/ -q
```
