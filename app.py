import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import tempfile

load_dotenv()

st.set_page_config(page_title="RAG Document Q&A", page_icon="📄")
st.title("📄 RAG-Powered Document Q&A")
st.caption("Upload any PDF and ask questions about it")

PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""Answer the question using ONLY the context below.
Always mention which part of the document supports your answer.

Context:
{context}

Question: {question}

Answer (with source reference):"""
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

@st.cache_resource
def get_embeddings():
    return FastEmbedEmbeddings()

def build_chain_from_pdf(pdf_file):
    # Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(pdf_file.read())
        tmp_path = f.name

    # Load and chunk
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # Embed and store in memory
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Build chain
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )
    return chain

# UI
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with st.spinner("Processing PDF..."):
        chain = build_chain_from_pdf(uploaded_file)
    st.success("PDF ready! Ask your questions below.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question about your document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chain.invoke(prompt)
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.info("👆 Upload a PDF to get started")