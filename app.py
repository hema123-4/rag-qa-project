import streamlit as st
from rag_chain import build_qa_chain

st.set_page_config(page_title="RAG Document Q&A", page_icon="📄")
st.title("📄 RAG-Powered Document Q&A")
st.caption("Ask questions about the Attention Is All You Need paper")

@st.cache_resource
def load_chain():
    chain, _ = build_qa_chain()
    return chain

chain = load_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about the document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chain.invoke(prompt)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})