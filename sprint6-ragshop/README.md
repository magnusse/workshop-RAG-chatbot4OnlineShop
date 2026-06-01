# Sprint 6 — Intent erkennen und Offer Request

Der Chatbot unterscheidet endlich zwischen *Produktfrage*, *Smalltalk* und *Angebotswunsch* — und kann Angebote tatsächlich erzeugen und an einen separaten Bounded Context (Order Management) übergeben.

## Abgedeckte Schritte der Domain Story

- **Schritt 4** — *Sales Assistant detects Intent (Product Info / Offer Request)*
- **Schritt 10** — *Sales Assistant creates & sends Offer Request*

Die Schritte 11–13 (Shopping Cart, Payment, Shipping) sind als Aufruf-Stub vorhanden — das Offer Request landet als JSON in `data/processed/offers/`. Eine echte Bezahl- und Versand-Anbindung ist Folgearbeit.

## Bausteine — und warum jetzt (Last Responsible Moment)

| Baustein | Begründung |
|---|---|
| `Intent` (Enum-VO) + `IntentDetector` (Domain Service) | Wir haben jetzt mehrere Bearbeitungspfade pro Frage — der Klassifikator wird zur expliziten Domain-Regel. LLM-basiert via `LLMPort`. |
| `Offer`, `OfferLine` (Value Objects) | Ein Offer ist mehr als eine Liste von Produkten — er gehört zu einem Kunden und enthält strukturierte Zeilen. |
| `OrderManagementPort` + `OfferReceipt` | Outbound-Port nach Order Management. Kapselt den anderen Bounded Context. |
| `order_management/` (eigener Bounded Context) | Eigene Domain: `OfferRequest`, `OrderService`, `JsonOfferStore`. Bewusst einfach gehalten — Hexagonal wäre Overkill für eine Schicht, die nur persistiert. |
| `OrderManagementAdapter` | Implementiert `OrderManagementPort`, ruft Order Management direkt auf. |
| `Conversation.recent_matches()` | Aggregate-Logik: aus den letzten Turns extrahieren, welche Produkte für ein Angebot in Frage kommen. |
| Offer-Extraction über 2. LLM-Call | Wenn der Customer ein Angebot will, extrahiert ein gezielter LLM-Call die gewünschten Produkt-IDs aus den Kandidaten der letzten Turns. |

## Vollständige Architektur (alle 3 Bounded Contexts)

```
ragshop/
├── composition_root.py             # einziger Wiring-Ort
├── product_information_management/ # BC 1 — einfach
├── sales_consultance/              # BC 2 — vollständig hexagonal + DDD
│   ├── domain/  (model, ports, services)
│   ├── application/  (api, service)
│   ├── infrastructure/  (LLM, PKB, conv-repo, OrderMgmt-Adapter, ApiKey)
│   └── interfaces/  (Gradio, PIM-Sink)
└── order_management/               # BC 3 — einfach
```

## Starten

```bash
cd sprint6-ragshop
pip install -r requirements.txt
python -m ragshop.sales_consultance.interfaces.gradio_chat_interface
```

## Tests

```bash
cd sprint6-ragshop
pytest tests/sales_consultance tests/product_information_management tests/order_management -q
```

## Wie der Offer-Flow sich im Chat anfühlt

1. *„Erzaehl mir vom EcoClean X200"* → PRODUCT_INFO → Retrieval, Antwort.
2. *„Und vom Z300?"* → PRODUCT_INFO → Query wird umgeschrieben, weiterer Retrieval.
3. *„Mach mir bitte ein Angebot fuer den X200"* → OFFER_REQUEST → IDs werden extrahiert, `Offer` gebaut, an Order Management gegeben, JSON-Datei landet in `data/processed/offers/<uuid>.json`.
