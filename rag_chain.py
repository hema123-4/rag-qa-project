from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from embeddings import load_vectorstore
from dotenv import load_dotenv
import os

load_dotenv()

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

def build_qa_chain():
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatGoogleGenerativeAI(model="models/gemini-3.1-flash-lite",
    google_api_key=os.environ["GOOGLE_API_KEY"],
    transport="rest")

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )
    return chain, retriever