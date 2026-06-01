# Sprint 5 — Product Updates und Retrieval-Filter

Die PKB ist jetzt ein lebendiger Datenstand: PIM kann zur Laufzeit Produkte hinzufügen, ändern oder löschen, und der Retrieval ignoriert veraltete oder als gelöscht markierte Einträge.

## Abgedeckte Schritte der Domain Story

Zwei Schritte werden verfeinert:

- **Schritt 2** *(erweitert)* — *PIM updates new/changed product INTO PKB*: jetzt auch zur Laufzeit, nicht nur beim Bootstrap.
- **Schritt 5** *(verfeinert)* — *select best content*: Retrieval filtert per Metadaten (`delflag`, `upddate`).

## Bausteine — und warum jetzt (Last Responsible Moment)

| Baustein | Begründung |
|---|---|
| `ProductCatalogService.add_or_update(product)` / `remove(id)` | Erstmals echte Schreibpfade in PIM. Eine Facade garantiert, dass jeder Catalog-Change **gleichzeitig** Master-Daten und PKB updatet — keine Inkonsistenzen. |
| `JsonProductRepository.save()` / `delete()` | Schreibseite des Master-Stores. Vorher reichte Lesen — jetzt nicht mehr. |
| `Product`-Schema erweitert (`source`, `upddate`, `delflag`, `prodcatversion`) | Die neuen Felder sind die Grundlage für Filter und Versionskontrolle. Sie werden jetzt eingeführt, weil sie jetzt verwendet werden — nicht früher. |
| `ProductKnowledgeSink.delete_product` / `ProductKnowledgePort.delete_product` | Symmetrie zum Upsert: ohne `delete` könnten nur neue Versionen entstehen, alte würden in der PKB hängen bleiben. |
| Retrieval `where`-Filter (`delflag == false` AND `upddate > now-365d`) | Step 5 verfeinert: nur aktuelle, nicht gelöschte Produkte werden vorgeschlagen. |
| Chunking schreibt jetzt Metadaten (`delflag`, `upddate`, `prodcatversion`) | Erst jetzt nötig — vorher gab es keinen Filter, der sie gebraucht hätte. |

## Was bewusst noch fehlt — und warum

- **Kein Intent-Routing.** Smalltalk, Produktfragen und Offer-Wünsche werden weiterhin gleich behandelt. Kommt in Sprint 6.
- **Kein Offer-Flow / kein Order-Management-Kontext.** Sprint 6.
- **Kein Domain Event für Catalog-Änderungen.** Die direkte Aufruf-Kopplung PIM → Sales Consultance reicht.

## Starten

```bash
cd sprint5-ragshop
pip install -r requirements.txt
python -m ragshop.sales_consultance.interfaces.gradio_chat_interface
```

## Tests

```bash
cd sprint5-ragshop
pytest tests/ -q
```
