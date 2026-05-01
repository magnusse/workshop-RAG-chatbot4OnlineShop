import json
import chromadb.utils

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
            "metadata": {"category": product["category"], "name": product["name"]}
        })
    return chunks

# 3. Vektordatenbank mit Chroma initialisieren
def setup_chroma(chunks, db_dir, embedding_model_name=EMBEDDING_MODEL_NAME, collection_name="products"):
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    ids = [chunk["id"] for chunk in chunks]

    client = PersistentClient(path=db_dir)

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model_name)

    # Bestehende oder neue Collection
    collection = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)

    # Daten hinzufügen
    collection.add(
        documents=texts,
        ids=ids,
        metadatas=metadatas
    )

    # Persistieren
    #client.persist()

    # Anzahl Einträge prüfen
    print("Entries in Collection:", len(collection.get()["ids"]))
    return collection

# Hauptfunktion
def main():
    print("Load Products...")
    products = load_products(DATA_PATH)
    print(f"{len(products)} Products loaded.")

    print("Chunking of products...")
    chunks = chunk_products(products)

    print("Save chunks in ChromaDB...")
    setup_chroma(chunks, DB_DIR)

    print("✅ Done: ChromaDB is ready for Retrieval.")

if __name__ == "__main__":
    main()