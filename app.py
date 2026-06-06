import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from dotenv import load_dotenv
import tempfile
from rag_chain import build_self_healing_rag

load_dotenv()

st.set_page_config(page_title="Self-Healing RAG Q&A", page_icon="🔁")
st.title("🔁 Self-Healing RAG Document Q&A")
st.caption("Upload any PDF — the AI critiques its own answers and retries if it hallucinated")

@st.cache_resource
def get_embeddings():
    return FastEmbedEmbeddings()

def build_graph_from_pdf(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(pdf_file.read())
        tmp_path = f.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    graph = build_self_healing_rag(retriever)
    return graph

# UI
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with st.spinner("Processing PDF and building Self-Healing RAG pipeline..."):
        graph = build_graph_from_pdf(uploaded_file)
    st.success("✅ PDF ready! Ask your questions below.")

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
            with st.spinner("Thinking... (critic agent evaluating answer)"):
                result = graph.invoke({
                    "question": prompt,
                    "reformulated_question": "",
                    "context": "",
                    "answer": "",
                    "is_grounded": False,
                    "retry_count": 0,
                    "final_answer": ""
                })
                response = result["final_answer"]
                grounded = result["is_grounded"]
                retries = result.get("retry_count", 0)

            st.markdown(response)

            # Show debug info
            if retries > 0:
                st.info(f"🔄 Query was reformulated {retries} time(s) before a grounded answer was found.")
            if grounded:
                st.success("✅ Answer verified as grounded in the document.")
            else:
                st.warning("⚠️ Could not fully verify answer against document context.")

        st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.info("👆 Upload a PDF to get started")