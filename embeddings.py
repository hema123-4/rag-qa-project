from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from ingest import load_and_chunk
from dotenv import load_dotenv

load_dotenv()

def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2"
    )

def create_vectorstore(pdf_path: str):
    print(f"Creating vectorstore for: {pdf_path}")
    chunks = load_and_chunk(pdf_path)
    print(f"Chunks created: {len(chunks)}")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory="./chroma_db"
    )
    print("ChromaDB created and saved!")
    return vectorstore

def load_vectorstore():
    return Chroma(
        persist_directory="./chroma_db",
        embedding_function=get_embeddings()
    )