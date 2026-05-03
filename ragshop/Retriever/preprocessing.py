import json
import chromadb.utils
from datetime import datetime

from chromadb.utils import embedding_functions
from chromadb import PersistentClient


# Pfade
# DATA_PATH = "../../data/raw/products.json"
DATA_PATH = "data/raw/products.json"
#DB_DIR = "../../vectorstore/chromadb"
DB_DIR = "vectorstore/chromadb"
COLLECTION_NAME = "products"
EMBEDDING_MODEL_NAME = "all-MiniLM-L12-v2"


# 1. Daten laden
def load_products(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# 2. Chunking: Pro Produkt ein Chunk aus Name + Beschreibung + Kompatibilität
def chunk_products(products):
    chunks = []
    for product in products:
        text = f"{product['name']} ({product['category']}): {product['description']}"
        if product.get("compatibility"):
            text += "\ncompatible with: " + ", ".join(product["compatibility"])
        chunks.append({
            "id": product["id"],
            "text": text,
            "metadata": {
                "category": product["category"],
                "name": product["name"],
                "price": product["price"],
                "source": product["source"],
                "upddate": int(datetime.strptime(product["upddate"], "%Y-%m-%d").timestamp()),
                "delflag": product["delflag"],
                "prodcatversion": product["prodcatversion"]
            }
        })
    return chunks


def _get_embedding_fn(model_name=EMBEDDING_MODEL_NAME):
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)


# 3a. Vektordatenbank komplett neu aufbauen (alle bestehenden Einträge werden gelöscht)
def rebuild_vectorstore(products, db_dir=DB_DIR, embedding_model_name=EMBEDDING_MODEL_NAME, collection_name=COLLECTION_NAME):
    client = PersistentClient(path=db_dir)

    try:
        client.delete_collection(name=collection_name)
        print(f"Bestehende Collection '{collection_name}' gelöscht.")
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        embedding_function=_get_embedding_fn(embedding_model_name)
    )

    chunks = chunk_products(products)
    collection.add(
        documents=[c["text"] for c in chunks],
        ids=[c["id"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks]
    )

    print(f"✅ Rebuild abgeschlossen. Einträge in Collection: {len(collection.get()['ids'])}")
    return collection


# 3b. Nur geänderte Einträge aktualisieren (Vergleich per prodcatversion)
def update_vectorstore(products, db_dir=DB_DIR, embedding_model_name=EMBEDDING_MODEL_NAME, collection_name=COLLECTION_NAME):
    client = PersistentClient(path=db_dir)
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=_get_embedding_fn(embedding_model_name)
    )

    # Bestehende Versionen aus ChromaDB laden
    existing = collection.get(include=["metadatas"])
    existing_versions = {
        eid: meta.get("prodcatversion")
        for eid, meta in zip(existing["ids"], existing["metadatas"])
    }

    # Nur Produkte mit geänderter oder fehlender Version selektieren
    changed = [
        p for p in products
        if p["id"] not in existing_versions
        or existing_versions[p["id"]] != p["prodcatversion"]
    ]

    if not changed:
        print("ℹ️ Keine Änderungen gefunden. VektorDB ist aktuell.")
        return collection

    chunks = chunk_products(changed)
    collection.upsert(
        documents=[c["text"] for c in chunks],
        ids=[c["id"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks]
    )

    print(f"✅ Update abgeschlossen. {len(changed)} Einträge aktualisiert. Gesamt: {len(collection.get()['ids'])}")
    return collection


# Hauptfunktion
def main():
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "rebuild"

    print("Lade Produkte...")
    products = load_products(DATA_PATH)
    print(f"{len(products)} Produkte geladen.")

    if mode == "update":
        print("Starte inkrementelles Update...")
        update_vectorstore(products)
    else:
        print("Starte vollständigen Rebuild...")
        rebuild_vectorstore(products)


if __name__ == "__main__":
    main()
