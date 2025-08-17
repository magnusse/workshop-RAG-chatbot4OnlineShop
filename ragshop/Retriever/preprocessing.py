import json

from chromadb.utils import embedding_functions
from chromadb import PersistentClient


# Pfade
DATA_PATH = "../../data/raw/products.json"
DB_DIR = "../../vectorstore/chromadb"
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
        text = f"{product['name']} ({product['kategorie']}): {product['beschreibung']}"
        if product.get("kompatibilitaet"):
            text += "\nKompatibel mit: " + ", ".join(product["kompatibilitaet"])
        chunks.append({
            "id": product["id"],
            "text": text,
            "metadata": {"kategorie": product["kategorie"], "name": product["name"]}
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
    print("Einträge in der Collection:", len(collection.get()["ids"]))
    return collection

# Hauptfunktion
def main():
    print("Lade Produkte...")
    products = load_products(DATA_PATH)
    print(f"{len(products)} Produkte geladen.")

    print("Chunking der Produkte...")
    chunks = chunk_products(products)

    print("Speichere Chunks mit LangChain in ChromaDB...")
    setup_chroma(chunks, DB_DIR)

    print("✅ Fertig: ChromaDB enthält jetzt die gechunkten Produktdaten.")

if __name__ == "__main__":
    main()