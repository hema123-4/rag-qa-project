from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_chunk(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(pages)
    print(f"Loaded {len(pages)} pages → {len(chunks)} chunks")
    return chunks

if __name__ == "__main__":
    chunks = load_and_chunk("sample.pdf")
    print(chunks[0].page_content)