from langchain_core.prompts import PromptTemplate

ANSWER_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""Answer the question using ONLY the context below.
Always mention which part of the document supports your answer.

Context:
{context}

Question: {question}

Answer (with source reference):"""
)

CRITIC_PROMPT = PromptTemplate(
    input_variables=["context", "answer"],
    template="""You are a strict fact-checker.

Given the context and the answer below, determine if the answer is fully grounded in the context.
Reply with ONLY 'YES' if the answer is grounded, or 'NO' if it contains information not in the context.

Context:
{context}

Answer:
{answer}

Is the answer grounded in the context? (YES/NO):"""
)

REFORMULATE_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""The previous search query did not retrieve enough relevant information.
Reformulate the following question to be more specific and retrieve better results.

Original question: {question}

Reformulated question:"""
)