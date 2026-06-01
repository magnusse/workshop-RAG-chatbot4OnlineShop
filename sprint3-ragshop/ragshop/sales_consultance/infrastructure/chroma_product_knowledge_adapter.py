from typing import List

from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

from ragshop.sales_consultance.domain.model.product_match import ProductMatch
from ragshop.sales_consultance.domain.ports.product_knowledge_port import (
    ProductKnowledgePort,
)


DB_DIR = "vectorstore/chromadb"
COLLECTION_NAME = "products"
EMBEDDING_MODEL_NAME = "all-MiniLM-L12-v2"


class ChromaProductKnowledgeAdapter(ProductKnowledgePort):
    """Persistent ChromaDB vector store. The chunking strategy lives here
    because *how* a product is indexed for retrieval is a Sales-Consultance
    concern, not a PIM one."""

    def __init__(
        self,
        db_dir: str = DB_DIR,
        collection_name: str = COLLECTION_NAME,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
    ):
        self._client = PersistentClient(db_dir)
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model_name
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name, embedding_function=embedding_fn
        )
        self._embedding_model = SentenceTransformer(embedding_model_name)

    # Domain Story Step 5: select best content from Product Knowledge Base
    def find_matches(self, query: str, k: int) -> List[ProductMatch]:
        embedding = self._embedding_model.encode(query)
        # Sprint 3: no metadata filter yet — every stored product is fair game.
        # Sprint 5 will add a delflag/upddate where-clause.
        results = self._collection.query(query_embeddings=[embedding], n_results=k)

        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        matches: List[ProductMatch] = []
        for pid, doc, meta in zip(ids, documents, metadatas):
            matches.append(
                ProductMatch(
                    product_id=pid,
                    name=meta.get("name", pid),
                    description=doc,
                    category=meta.get("category"),
                    price=meta.get("price"),
                )
            )
        return matches

    # Domain Story Step 2: PIM updates new/changed product INTO Product Knowledge Base
    def upsert_product(self, product: dict) -> None:
        chunk = self._build_chunk(product)
        self._collection.upsert(
            documents=[chunk["text"]],
            ids=[chunk["id"]],
            metadatas=[chunk["metadata"]],
        )

    @staticmethod
    def _build_chunk(product: dict) -> dict:
        text = f"{product['name']} ({product['category']}): {product['description']}"
        if product.get("compatibility"):
            text += "\ncompatible with: " + ", ".join(product["compatibility"])
        # Sprint 3 metadata: only what's needed to render a ProductMatch.
        # Sprint 5 will add delflag, upddate, prodcatversion for filtering.
        return {
            "id": product["id"],
            "text": text,
            "metadata": {
                "name": product["name"],
                "category": product["category"],
                "price": product["price"],
            },
        }
