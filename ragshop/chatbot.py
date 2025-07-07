from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import os
from ragshop.CustomLLM import WPSCustomLLM

# Einstellungen
DB_DIR = "../vectorstore/chromadb"
COLLECTION_NAME = "products"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
OPENAI_MODEL_NAME = "gpt-3.5-turbo"  # alternativ: "gpt-4"

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings, collection_name=COLLECTION_NAME)
    return db

def build_qa_chain(db):
    retriever = db.as_retriever(search_kwargs={"k": 3})
    token = os.getenv("WEBUI_API_KEY")
    llm = WPSCustomLLM(api_key=token)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

def chat():
    print("🛍️ Willkommen beim Produkt-Chatbot. Stelle deine Frage oder tippe 'exit':\n")
    vectorstore = load_vectorstore()
    qa_chain = build_qa_chain(vectorstore)

    while True:
        query = input("Du: ")
        if query.lower() in ("exit", "quit"):
            break
        result = qa_chain({"query": query})
        print("\nAntwort:")
        print(result["result"])
        print("\n---\n")

if __name__ == "__main__":
    if not os.getenv("WEBUI_API_KEY"):
        print("❌ Bitte setze die Umgebungsvariable WEBUI_API_KEY.")
    else:
        token = os.getenv("WEBUI_API_KEY")
        chat()