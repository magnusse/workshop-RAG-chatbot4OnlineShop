import json

#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma



# Pfade
DATA_PATH = "../data/raw/products.json"
DB_DIR = "../vectorstore/chromadb"

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
def setup_chroma(chunks, db_dir, embedding_model_name="all-MiniLM-L6-v2", collection_name="products"):
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    ids = [chunk["id"] for chunk in chunks]

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    db = Chroma.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas, ids=ids, persist_directory=db_dir, collection_name=collection_name)
    db.persist()
    return db

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