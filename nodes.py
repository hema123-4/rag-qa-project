from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from prompts import ANSWER_PROMPT, CRITIC_PROMPT, REFORMULATE_PROMPT
from state import RAGState
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

def retrieve_node(retriever):
    def retrieve(state: RAGState) -> RAGState:
        q = state.get("reformulated_question") or state["question"]
        docs = retriever.invoke(q)
        context = "\n\n".join(doc.page_content for doc in docs)
        return {**state, "context": context}
    return retrieve

def generate_node(state: RAGState) -> RAGState:
    q = state.get("reformulated_question") or state["question"]
    chain = ANSWER_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"context": state["context"], "question": q})
    return {**state, "answer": answer}

def critic_node(state: RAGState) -> RAGState:
    chain = CRITIC_PROMPT | llm | StrOutputParser()
    result = chain.invoke({"context": state["context"], "answer": state["answer"]})
    is_grounded = "YES" in result.upper()
    return {**state, "is_grounded": is_grounded}

def reformulate_node(state: RAGState) -> RAGState:
    chain = REFORMULATE_PROMPT | llm | StrOutputParser()
    new_question = chain.invoke({"question": state["question"]})
    retry_count = state.get("retry_count", 0) + 1
    return {**state, "reformulated_question": new_question, "retry_count": retry_count}

def finalize_node(state: RAGState) -> RAGState:
    if state["is_grounded"]:
        return {**state, "final_answer": state["answer"]}
    else:
        return {**state, "final_answer": "I don't have enough information in the document to answer this accurately."}

def should_retry(state: RAGState) -> str:
    if state["is_grounded"]:
        return "finalize"
    elif state.get("retry_count", 0) >= 2:
        return "finalize"
    else:
        return "reformulate"