# Sprint 2 — Chatbot über LLM

Der Chatbot generiert echte Antworten über das WPS-LLM. Noch kein RAG, keine Historie, kein Intent — nur Frage rein, LLM-Antwort raus.

## Abgedeckte Schritte der Domain Story

Zusätzlich zu Sprint 1:

- **Schritt 6** — *generate prompt AND send to LLM*
- **Schritt 7** — *LLM responds with Answer*

## Bausteine — und warum jetzt (Last Responsible Moment)

| Baustein | Begründung |
|---|---|
| `LLMPort` (Outbound-Port) | Erst jetzt sinnvoll: wir wollen den `SalesConsultantService` testen, ohne den realen WPS-Endpoint zu rufen. Ohne Port wäre der Service nicht isoliert testbar. |
| `WpsLLMAdapter` | Konkrete Implementierung des `LLMPort` gegen `https://gpt.wps.de/api/chat/completions`. |
| `api_key_provider` (Keyring + Hybrid) | Wir brauchen einen API-Key. Statt `os.getenv` durchs ganze System zu schleifen, kapseln wir die Beschaffung — keyring → ENV → Terminal-Prompt. |
| `composition_root.py` | Erste Stelle, an der mehrere Klassen verkabelt werden müssen (Provider → Adapter → Service). Davor gab's nichts zu verkabeln. |
| `tests/.../fakes/fake_llm.py` | Beweis, dass der Port seinen Zweck erfüllt: Tests laufen ohne Netz. |

## Was bewusst noch fehlt — und warum

- **Keine Produkt-/Vektordatenbank.** Das LLM antwortet aus seinem Trainingswissen — noch nicht eingeschränkt auf unser Sortiment. Das motiviert Sprint 3 (RAG).
- **Keine Konversations-Historie.** Jede Frage wird unabhängig beantwortet. Folgefragen wie *„Und wie viel kostet der?"* funktionieren noch nicht. Kommt in Sprint 4.
- **Kein Intent-Routing.** Wir nehmen an, alle Fragen sind Produktfragen — kein Smalltalk-/Offer-Unterschied. Kommt in Sprint 6.
- **Keine `Question`-/`Answer`-VOs.** Plain `str` ist noch ausreichend; eine VO würde nichts kapseln.

## Vorbereitung

Beim ersten Start fragt der Provider nach dem WPS-API-Key (verdeckter Terminal-Prompt) und speichert ihn im OS-Keyring. Folgeläufe lesen ihn dort stumm aus.

Alternativ einmalig per Umgebungsvariable:
```bash
export WEBUI_API_KEY="dein-key"
```
Der Key wird beim ersten Lauf in den Keyring übernommen.

Reset (z.B. nach Ablauf):
```bash
WEBUI_API_KEY_RESET=1 python -m ragshop.sales_consultance.interfaces.gradio_chat_interface
```

## Starten

```bash
cd sprint2-ragshop
pip install -r requirements.txt
python -m ragshop.sales_consultance.interfaces.gradio_chat_interface
```

## Tests

```bash
cd sprint2-ragshop
pytest tests/ -q
```
